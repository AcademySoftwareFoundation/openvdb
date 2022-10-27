BeginTestSection["Image Generic Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBImage3D]
	,
	"Index"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBImage3D]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBImage3D]
	,
	{Resampling->Automatic, "ScalingFactor"->1.}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBImage3D]
	,
	{"ArgumentsPattern"->{_, _., OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBImage3D[], OpenVDBLink`OpenVDBImage3D["error"], OpenVDBLink`OpenVDBImage3D[bmr, "error"], OpenVDBLink`OpenVDBImage3D[bmr, {{0, 1}, {0, 1}, {0, 1}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBImage3D::argt, OpenVDBImage3D::grid2, OpenVDBImage3D::bbox3d, OpenVDBImage3D::nonopt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBImage3D[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBImage3D::npxl2, OpenVDBImage3D::npxl2, OpenVDBImage3D::npxl2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDepthImage]
	,
	"Index"	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBDepthImage]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBDepthImage]
	,
	{}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDepthImage]
	,
	{"ArgumentsPattern"->{_, _., _., _., OptionsPattern[]}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBDepthImage[], OpenVDBLink`OpenVDBDepthImage["error"], OpenVDBLink`OpenVDBDepthImage[bmr, "error"], OpenVDBLink`OpenVDBDepthImage[bmr, Automatic, "error"], OpenVDBLink`OpenVDBDepthImage[bmr, Automatic, 2., "error"], OpenVDBLink`OpenVDBDepthImage[bmr, Automatic, 2., {0, 1}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDepthImage::argb, OpenVDBDepthImage::grid2, OpenVDBDepthImage::bbox3d, OpenVDBDepthImage::gamma, OpenVDBDepthImage::range, OpenVDBDepthImage::argb}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBDepthImage[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDepthImage::npxl2, OpenVDBDepthImage::npxl2, OpenVDBDepthImage::npxl2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBProjectionImage]
	,
	"Index"	
]

VerificationTest[(* 15 *)
	Attributes[OpenVDBLink`OpenVDBProjectionImage]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 16 *)
	Options[OpenVDBLink`OpenVDBProjectionImage]
	,
	{}	
]

VerificationTest[(* 17 *)
	SyntaxInformation[OpenVDBLink`OpenVDBProjectionImage]
	,
	{"ArgumentsPattern"->{_, _., OptionsPattern[]}}	
]

VerificationTest[(* 18 *)
	{OpenVDBLink`OpenVDBProjectionImage[], OpenVDBLink`OpenVDBProjectionImage["error"], OpenVDBLink`OpenVDBProjectionImage[bmr, "error"], OpenVDBLink`OpenVDBProjectionImage[bmr, Automatic, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBProjectionImage::argt, OpenVDBProjectionImage::grid2, OpenVDBProjectionImage::bbox3d, OpenVDBProjectionImage::argt}
]

VerificationTest[(* 19 *)
	(OpenVDBLink`OpenVDBProjectionImage[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBProjectionImage::npxl2, OpenVDBProjectionImage::npxl2, OpenVDBProjectionImage::npxl2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSliceImage]
	,
	"Index"	
]

VerificationTest[(* 21 *)
	Attributes[OpenVDBLink`OpenVDBSliceImage]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 22 *)
	Options[OpenVDBLink`OpenVDBSliceImage]
	,
	{"MirrorSlice"->False}	
]

VerificationTest[(* 23 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSliceImage]
	,
	{"ArgumentsPattern"->{_, _, _., OptionsPattern[]}}	
]

VerificationTest[(* 24 *)
	{OpenVDBLink`OpenVDBSliceImage[], OpenVDBLink`OpenVDBSliceImage["error"], OpenVDBLink`OpenVDBSliceImage[bmr, "error"], OpenVDBLink`OpenVDBSliceImage[bmr, 0, "error"], OpenVDBLink`OpenVDBSliceImage[bmr, 0, {{0, 1}, {0, 1}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSliceImage::argb, OpenVDBSliceImage::grid2, OpenVDBSliceImage::zslice, OpenVDBSliceImage::bbox2d, OpenVDBSliceImage::nonopt}
]

VerificationTest[(* 25 *)
	(OpenVDBLink`OpenVDBSliceImage[OpenVDBLink`OpenVDBCreateGrid[1., #1], 0, {{0, 1}, {0, 1}, {0, 1}}]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean",    "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSliceImage::npxl2, OpenVDBSliceImage::npxl2, OpenVDBSliceImage::npxl2, General::stop, OpenVDBSliceImage::bbox2d, OpenVDBSliceImage::bbox2d}
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDynamicSliceImage]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 27 *)
	Attributes[OpenVDBLink`OpenVDBDynamicSliceImage]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 28 *)
	Options[OpenVDBLink`OpenVDBDynamicSliceImage]
	,
	{DisplayFunction->Identity, ImageSize->Automatic}	
]

VerificationTest[(* 29 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDynamicSliceImage]
	,
	{"ArgumentsPattern"->{_, OptionsPattern[]}}	
]

VerificationTest[(* 30 *)
	{OpenVDBLink`OpenVDBDynamicSliceImage[], OpenVDBLink`OpenVDBDynamicSliceImage["error"], OpenVDBLink`OpenVDBDynamicSliceImage[bmr, "error"]}
	,
	{$Failed, $Failed, $Failed}
	,
	{OpenVDBDynamicSliceImage::argx, OpenVDBDynamicSliceImage::grid2, OpenVDBDynamicSliceImage::nonopt}
]

VerificationTest[(* 31 *)
	(OpenVDBLink`OpenVDBDynamicSliceImage[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDynamicSliceImage::npxl2, OpenVDBDynamicSliceImage::npxl2, OpenVDBDynamicSliceImage::npxl2, General::stop, OpenVDBDynamicSliceImage::empty, OpenVDBDynamicSliceImage::empty}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 32 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb],    OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 33 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 34 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 35 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 36 *)
	im=OpenVDBLink`OpenVDBImage3D[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25, 34}, 0.10850702392570608}	
]

VerificationTest[(* 37 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31, 31}, 0.07659452354672187}	
]

VerificationTest[(* 38 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 31}, 0.11072525112195607}	
]

VerificationTest[(* 39 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 24}, 0.14302011603252657}	
]

VerificationTest[(* 40 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19, 24}, 0.1036027096398614}	
]

VerificationTest[(* 41 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11, 14}, 0.2601407505150823}	
]

VerificationTest[(* 42 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, "ScalingFactor"->1.3, Resampling->"Nearest"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.14423111579175063}	
]

VerificationTest[(* 43 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, "ScalingFactor"->1.3, Resampling->"Linear"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.19102034905472848}	
]

VerificationTest[(* 44 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, "ScalingFactor"->1.3, Resampling->"Quadratic"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.22142316507699136}	
]

VerificationTest[(* 45 *)
	im=OpenVDBLink`OpenVDBImage3D[bmr, "ScalingFactor"->0.5];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {43, 17, 21}, 0.1301091708913388}	
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 46 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 47 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 48 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 49 *)
	im=OpenVDBLink`OpenVDBDepthImage[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25}, 0.26463413836041955}	
]

VerificationTest[(* 50 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.2444580956192592}	
]

VerificationTest[(* 51 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.35676031767035876}	
]

VerificationTest[(* 52 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4031068659831016}	
]

VerificationTest[(* 53 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19}, 0.2537302976919089}	
]

VerificationTest[(* 54 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.8025942667702998}	
]

VerificationTest[(* 55 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, {{0, 10}, {0, 10}, {-3, 10}}->"World", 2.53];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {101, 101}, 0.014254320773315505}	
]

VerificationTest[(* 56 *)
	im=OpenVDBLink`OpenVDBDepthImage[bmr, Automatic, 2., {0.1, 1}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4101067692806412}	
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 57 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 58 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 59 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 60 *)
	im=OpenVDBLink`OpenVDBProjectionImage[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25}, 0.3186204122674711}	
]

VerificationTest[(* 61 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.3506763787721124}	
]

VerificationTest[(* 62 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5293898809523809}	
]

VerificationTest[(* 63 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5293898809523809}	
]

VerificationTest[(* 64 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19}, 0.3562753036437247}	
]

VerificationTest[(* 65 *)
	im=OpenVDBLink`OpenVDBProjectionImage[bmr, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.9090909090909091}	
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 66 *)
	im=OpenVDBLink`OpenVDBSliceImage[bmr, 0];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19166812558356686}	
]

VerificationTest[(* 67 *)
	im=OpenVDBLink`OpenVDBSliceImage[bmr, 4->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19799544817927173}	
]

VerificationTest[(* 68 *)
	im=OpenVDBLink`OpenVDBSliceImage[bmr, 0.5->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19392507002801124}	
]

VerificationTest[(* 69 *)
	im=OpenVDBLink`OpenVDBSliceImage[bmr, 0.5->"World", {{-0.5, 1.3}, {-0.5, 2.3}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {19, 29}, 0.11333404505177759}	
]

VerificationTest[(* 70 *)
	im=OpenVDBLink`OpenVDBSliceImage[bmr, 5->"Index", {{-20, 20}, {-10, 10}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {41, 21}, 0.26237616998018753}	
]

VerificationTest[(* 71 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[bmr, 5->"Index", {{0, 1}, {8, 9}}]]
	,
	{{0.6274509803921569, 0.6431372549019608}, {0.4627450980392157, 0.47843137254901963}}	
]

VerificationTest[(* 72 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[bmr, 5->"Index", {{0, 1}, {8, 9}}, "MirrorSlice"->True]]
	,
	{{0.6431372549019608, 0.6274509803921569}, {0.47843137254901963, 0.4627450980392157}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 73 *)
	OpenVDBLink`OpenVDBDynamicSliceImage[bmr]
	,
	$Failed	
]

VerificationTest[(* 74 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[vdb]]
	,
	DynamicModule	
]

VerificationTest[(* 75 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[fog]]
	,
	DynamicModule	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 76 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty],    OpenVDBLink`OpenVDBScalarGridQ[vdb],OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 77 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 78 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 79 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 41}, 0.14580689491242496}	
]

VerificationTest[(* 80 *)
	im=OpenVDBLink`OpenVDBImage3D[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25, 34}, 0.10850702392570608}	
]

VerificationTest[(* 81 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31, 31}, 0.07659452354672187}	
]

VerificationTest[(* 82 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 31}, 0.11072525112195607}	
]

VerificationTest[(* 83 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32, 24}, 0.14302011603252657}	
]

VerificationTest[(* 84 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19, 24}, 0.1036027096398614}	
]

VerificationTest[(* 85 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11, 14}, 0.2601407505150823}	
]

VerificationTest[(* 86 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Nearest"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.14423111579175063}	
]

VerificationTest[(* 87 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Linear"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.19241374929752142}	
]

VerificationTest[(* 88 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Quadratic"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {109, 42, 53}, 0.22224669222600837}	
]

VerificationTest[(* 89 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->0.5];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {43, 17, 21}, 0.1301091708913388}	
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 90 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 91 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 92 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4955575923834528}	
]

VerificationTest[(* 93 *)
	im=OpenVDBLink`OpenVDBDepthImage[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25}, 0.2646341382781378}	
]

VerificationTest[(* 94 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.2444580956192592}	
]

VerificationTest[(* 95 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.35676031767035876}	
]

VerificationTest[(* 96 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4031068659831016}	
]

VerificationTest[(* 97 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19}, 0.2537302976919089}	
]

VerificationTest[(* 98 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.8025942667702998}	
]

VerificationTest[(* 99 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"World", 2.53];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {101, 101}, 0.014254320773315505}	
]

VerificationTest[(* 100 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic, 2., {0.1, 1}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.4101067692806412}	
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 101 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 102 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 103 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5658482142857143}	
]

VerificationTest[(* 104 *)
	im=OpenVDBLink`OpenVDBProjectionImage[fog];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {78, 25}, 0.3186204122674711}	
]

VerificationTest[(* 105 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.3506763787721124}	
]

VerificationTest[(* 106 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5293898809523809}	
]

VerificationTest[(* 107 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.5293898809523809}	
]

VerificationTest[(* 108 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 19}, 0.3562753036437247}	
]

VerificationTest[(* 109 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.9090909090909091}	
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 110 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19166812558356686}	
]

VerificationTest[(* 111 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 4->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19799544817927173}	
]

VerificationTest[(* 112 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {84, 32}, 0.19392507002801124}	
]

VerificationTest[(* 113 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World", {{-0.5, 1.3}, {-0.5, 2.3}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {19, 29}, 0.11333404505177759}	
]

VerificationTest[(* 114 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{-20, 20}, {-10, 10}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {41, 21}, 0.26237616998018753}	
]

VerificationTest[(* 115 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{0, 1}, {8, 9}}]]
	,
	{{0.6274509803921569, 0.6431372549019608}, {0.4627450980392157, 0.47843137254901963}}	
]

VerificationTest[(* 116 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{0, 1}, {8, 9}}, "MirrorSlice"->True]]
	,
	{{0.6431372549019608, 0.6274509803921569}, {0.47843137254901963, 0.4627450980392157}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 117 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[vdb]]
	,
	DynamicModule	
]

VerificationTest[(* 118 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[fog]]
	,
	DynamicModule	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Byte"]

BeginTestSection["Initialization"]

VerificationTest[(* 119 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Byte"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[25*i, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[25*i, {i, 10}]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 120 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.010784313725490196}	
]

VerificationTest[(* 121 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.010784313725490196}	
]

VerificationTest[(* 122 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.010784313725490196}	
]

VerificationTest[(* 123 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31, 31}, 0.0003619990509043065}	
]

VerificationTest[(* 124 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 31}, 0.0034788108791903856}	
]

VerificationTest[(* 125 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 3}, 0.00196078431372549}	
]

VerificationTest[(* 126 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3, 3}, 0.0054466230936819175}	
]

VerificationTest[(* 127 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11, 14}, 0.0063661828367710714}	
]

VerificationTest[(* 128 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Nearest"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.012673253188394146}	
]

VerificationTest[(* 129 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Linear"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.010185011646898159}	
]

VerificationTest[(* 130 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Quadratic"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.08913402411488036}	
]

VerificationTest[(* 131 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->0.5];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {6, 6, 6}, 0.013616557734204794}	
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 132 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 5.050818592309952}	
]

VerificationTest[(* 133 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 5.050818592309952}	
]

VerificationTest[(* 134 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 5.050818592309952}	
]

VerificationTest[(* 135 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.004487469156624499}	
]

VerificationTest[(* 136 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.04312457859516144}	
]

VerificationTest[(* 137 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 1.0014139418303967}	
]

VerificationTest[(* 138 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.011782848586638769}	
]

VerificationTest[(* 139 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 4.176405082556827}	
]

VerificationTest[(* 140 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"World", 2.53];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 4.171260416507721}	
]

VerificationTest[(* 141 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic, 2., {0.1, 1}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 5.0402313220500945}	
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 142 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 143 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 144 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 145 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.01040582726326743}	
]

VerificationTest[(* 146 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 147 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.02}	
]

VerificationTest[(* 148 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.}	
]

VerificationTest[(* 149 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.08264462809917356}	
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 150 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 151 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 4->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.00784313725490196}	
]

VerificationTest[(* 152 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 153 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World", {{-0.5, 1.3}, {-0.5, 2.3}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.}	
]

VerificationTest[(* 154 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{-20, 20}, {-10, 10}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {41, 21}, 0.0011386668488533624}	
]

VerificationTest[(* 155 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}]]
	,
	{{0.49019607843137253, 0.}, {0., 0.}}	
]

VerificationTest[(* 156 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}, "MirrorSlice"->True]]
	,
	{{0., 0.49019607843137253}, {0., 0.}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 157 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[vdb]]
	,
	DynamicModule	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 158 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 159 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 160 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 161 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 162 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31, 31}, 0.0004028062166426102}	
]

VerificationTest[(* 163 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 31}, 0.003870967741935484}	
]

VerificationTest[(* 164 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 3}, 0.01}	
]

VerificationTest[(* 165 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3, 3}, 0.05555555555555555}	
]

VerificationTest[(* 166 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11, 14}, 0.0070838252656434475}	
]

VerificationTest[(* 167 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Nearest"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.01729631315430132}	
]

VerificationTest[(* 168 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Linear"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.07282658170232134}	
]

VerificationTest[(* 169 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Quadratic"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.19253527537551207}	
]

VerificationTest[(* 170 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->0.5];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {6, 6, 6}, 0.018518518518518517}	
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 171 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08827828764915466}	
]

VerificationTest[(* 172 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08827828764915466}	
]

VerificationTest[(* 173 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08827828764915466}	
]

VerificationTest[(* 174 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.005644630132927235}	
]

VerificationTest[(* 175 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.05424489557743072}	
]

VerificationTest[(* 176 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.034422205686569216}	
]

VerificationTest[(* 177 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.12018504738807678}	
]

VerificationTest[(* 178 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.07615359262986617}	
]

VerificationTest[(* 179 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"World", 2.53];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.0687350152937834}	
]

VerificationTest[(* 180 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic, 2., {0.1, 1}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.07307341456413269}	
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 181 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 182 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 183 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 184 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.01040582726326743}	
]

VerificationTest[(* 185 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.1}	
]

VerificationTest[(* 186 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.04}	
]

VerificationTest[(* 187 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.16666666666666666}	
]

VerificationTest[(* 188 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.08264462809917356}	
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 189 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 190 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 4->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.02}	
]

VerificationTest[(* 191 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 192 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World", {{-0.5, 1.3}, {-0.5, 2.3}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.}	
]

VerificationTest[(* 193 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{-20, 20}, {-10, 10}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {41, 21}, 0.0011614401858304297}	
]

VerificationTest[(* 194 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}]]
	,
	{{1, 0}, {0, 0}}	
]

VerificationTest[(* 195 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}, "MirrorSlice"->True]]
	,
	{{0, 1}, {0, 0}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 196 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[vdb]]
	,
	DynamicModule	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 197 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBImage3D"]

VerificationTest[(* 198 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 199 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 200 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 10}, 0.012}	
]

VerificationTest[(* 201 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31, 31}, 0.0004028062166426102}	
]

VerificationTest[(* 202 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 31}, 0.003870967741935484}	
]

VerificationTest[(* 203 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10, 3}, 0.01}	
]

VerificationTest[(* 204 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3, 3}, 0.05555555555555555}	
]

VerificationTest[(* 205 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11, 14}, 0.0070838252656434475}	
]

VerificationTest[(* 206 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Nearest"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.018206645425580335}	
]

VerificationTest[(* 207 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Linear"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.07373691397360037}	
]

VerificationTest[(* 208 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->1.3, Resampling->"Quadratic"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {13, 13, 13}, 0.19344560764679108}	
]

VerificationTest[(* 209 *)
	im=OpenVDBLink`OpenVDBImage3D[vdb, "ScalingFactor"->0.5];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {6, 6, 6}, 0.018518518518518517}	
]

EndTestSection[]

BeginTestSection["OpenVDBDepthImage"]

VerificationTest[(* 210 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.0604760779440403}	
]

VerificationTest[(* 211 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.0604760779440403}	
]

VerificationTest[(* 212 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.0604760779440403}	
]

VerificationTest[(* 213 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.004065708271794711}	
]

VerificationTest[(* 214 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.039071456491947175}	
]

VerificationTest[(* 215 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.024422205686569214}	
]

VerificationTest[(* 216 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.12018504738807678}	
]

VerificationTest[(* 217 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.05634485148201304}	
]

VerificationTest[(* 218 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"World", 2.53];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.04800332799430721}	
]

VerificationTest[(* 219 *)
	im=OpenVDBLink`OpenVDBDepthImage[vdb, Automatic, 2., {0.1, 1}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.04634328879415989}	
]

EndTestSection[]

BeginTestSection["OpenVDBProjectionImage"]

VerificationTest[(* 220 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08}	
]

VerificationTest[(* 221 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, Automatic];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08}	
]

VerificationTest[(* 222 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08}	
]

VerificationTest[(* 223 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 30}, {0, 30}, {0, 30}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {31, 31}, 0.008324661810613945}	
]

VerificationTest[(* 224 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 30}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.08}	
]

VerificationTest[(* 225 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {0, 2.3}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.03}	
]

VerificationTest[(* 226 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0.1, 1.3}, {0.47, 2.3}, {0, 2.3}}->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.16666666666666666}	
]

VerificationTest[(* 227 *)
	im=OpenVDBLink`OpenVDBProjectionImage[vdb, {{0, 10}, {0, 10}, {-3, 10}}->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {11, 11}, 0.06611570247933884}	
]

EndTestSection[]

BeginTestSection["OpenVDBSliceImage"]

VerificationTest[(* 228 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 229 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 4->"Index"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.01}	
]

VerificationTest[(* 230 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World"];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {10, 10}, 0.}	
]

VerificationTest[(* 231 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 0.5->"World", {{-0.5, 1.3}, {-0.5, 2.3}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {2, 3}, 0.}	
]

VerificationTest[(* 232 *)
	im=OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{-20, 20}, {-10, 10}}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {41, 21}, 0.0023228803716608595}	
]

VerificationTest[(* 233 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}]]
	,
	{{1, 0}, {0, 0}}	
]

VerificationTest[(* 234 *)
	ImageData[OpenVDBLink`OpenVDBSliceImage[vdb, 5->"Index", {{5, 6}, {4, 5}}, "MirrorSlice"->True]]
	,
	{{0, 1}, {0, 0}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDynamicSliceImage"]

VerificationTest[(* 235 *)
	Head[OpenVDBLink`OpenVDBDynamicSliceImage[vdb]]
	,
	DynamicModule	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
