BeginTestSection["DistanceMeasure Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBMember"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBMember]
	,
	"World"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBMember]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBMember]
	,
	{"IsoValue"->Automatic}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBMember]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBMember[], OpenVDBLink`OpenVDBMember["error"], OpenVDBLink`OpenVDBMember[bmr], OpenVDBLink`OpenVDBMember[bmr, "error"], OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0, 0}}], OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}}->"error"], OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBMember::argrx, OpenVDBMember::argr, OpenVDBMember::argr, OpenVDBMember::coord, OpenVDBMember::coord, OpenVDBMember::gridspace, OpenVDBMember::nonopt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBMember[OpenVDBLink`OpenVDBCreateGrid[1., #1], {0, 0, 0}]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBMember::scalargrid2, OpenVDBMember::scalargrid2, OpenVDBMember::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBNearest"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBNearest]
	,
	"World"	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBNearest]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBNearest]
	,
	{"IsoValue"->Automatic}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBNearest]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBNearest[], OpenVDBLink`OpenVDBNearest["error"], OpenVDBLink`OpenVDBNearest[bmr], OpenVDBLink`OpenVDBNearest[bmr, "error"], OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0, 0}}], OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}}->"error"], OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBNearest::argrx, OpenVDBNearest::argr, OpenVDBNearest::argr, OpenVDBNearest::coord, OpenVDBNearest::coord, OpenVDBNearest::gridspace, OpenVDBNearest::nonopt}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBNearest[OpenVDBLink`OpenVDBCreateGrid[1., #1], {0, 0, 0}]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBNearest::scalargrid2, OpenVDBNearest::scalargrid2, OpenVDBNearest::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBDistance"]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDistance]
	,
	"World"	
]

VerificationTest[(* 15 *)
	Attributes[OpenVDBLink`OpenVDBDistance]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 16 *)
	Options[OpenVDBLink`OpenVDBDistance]
	,
	{"IsoValue"->Automatic}	
]

VerificationTest[(* 17 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDistance]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 18 *)
	{OpenVDBLink`OpenVDBDistance[], OpenVDBLink`OpenVDBDistance["error"], OpenVDBLink`OpenVDBDistance[bmr], OpenVDBLink`OpenVDBDistance[bmr, "error"], OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0, 0}}], OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}}->"error"], OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDistance::argrx, OpenVDBDistance::argr, OpenVDBDistance::argr, OpenVDBDistance::coord, OpenVDBDistance::coord, OpenVDBDistance::gridspace, OpenVDBDistance::nonopt}
]

VerificationTest[(* 19 *)
	(OpenVDBLink`OpenVDBDistance[OpenVDBLink`OpenVDBCreateGrid[1., #1], {0, 0, 0}]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDistance::scalargrid2, OpenVDBDistance::scalargrid2, OpenVDBDistance::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBSignedDistance"]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSignedDistance]
	,
	"World"	
]

VerificationTest[(* 21 *)
	Attributes[OpenVDBLink`OpenVDBSignedDistance]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 22 *)
	Options[OpenVDBLink`OpenVDBSignedDistance]
	,
	{"IsoValue"->Automatic}	
]

VerificationTest[(* 23 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSignedDistance]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 24 *)
	{OpenVDBLink`OpenVDBSignedDistance[], OpenVDBLink`OpenVDBSignedDistance["error"], OpenVDBLink`OpenVDBSignedDistance[bmr], OpenVDBLink`OpenVDBSignedDistance[bmr, "error"], OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0, 0}}], OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}}->"error"], OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSignedDistance::argrx, OpenVDBSignedDistance::argr, OpenVDBSignedDistance::argr, OpenVDBSignedDistance::coord, OpenVDBSignedDistance::coord, OpenVDBSignedDistance::gridspace, OpenVDBSignedDistance::nonopt}
]

VerificationTest[(* 25 *)
	(OpenVDBLink`OpenVDBSignedDistance[OpenVDBLink`OpenVDBCreateGrid[1., #1], {0, 0, 0}]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSignedDistance::scalargrid2, OpenVDBSignedDistance::scalargrid2, OpenVDBSignedDistance::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBFillWithBalls"]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBFillWithBalls]
	,
	"World"	
]

VerificationTest[(* 27 *)
	Attributes[OpenVDBLink`OpenVDBFillWithBalls]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 28 *)
	Options[OpenVDBLink`OpenVDBFillWithBalls]
	,
	{"IsoValue"->Automatic, "Overlapping"->False, "ReturnType"->Automatic, "SeedCount"->Automatic}	
]

VerificationTest[(* 29 *)
	SyntaxInformation[OpenVDBLink`OpenVDBFillWithBalls]
	,
	{"ArgumentsPattern"->{_, _, _., OptionsPattern[]}}	
]

VerificationTest[(* 30 *)
	{OpenVDBLink`OpenVDBFillWithBalls[], OpenVDBLink`OpenVDBFillWithBalls["error"], OpenVDBLink`OpenVDBFillWithBalls[bmr], OpenVDBLink`OpenVDBFillWithBalls[bmr, "error"], OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, "error"], OpenVDBLink`OpenVDBSignedDistance[bmr, 10, 1->"error"], OpenVDBLink`OpenVDBSignedDistance[bmr, 10, 3->"World", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFillWithBalls::argt, OpenVDBFillWithBalls::argtu, OpenVDBFillWithBalls::argtu, OpenVDBFillWithBalls::intpm, OpenVDBFillWithBalls::rspec, OpenVDBSignedDistance::optrs, OpenVDBSignedDistance::nonopt}
]

VerificationTest[(* 31 *)
	(OpenVDBLink`OpenVDBFillWithBalls[OpenVDBLink`OpenVDBCreateGrid[1., #1], 10, 3]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFillWithBalls::scalargrid2, OpenVDBFillWithBalls::scalargrid2, OpenVDBFillWithBalls::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 32 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];maxcoord=MaximalBy[MeshCoordinates[bmr], #1[[2]]&][[1]];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBMember"]

VerificationTest[(* 33 *)
	OpenVDBLink`OpenVDBMember[bmr, {0, 0, 0}]
	,
	1	
]

VerificationTest[(* 34 *)
	OpenVDBLink`OpenVDBMember[bmr, {0, 0, 0}->"World"]
	,
	1	
]

VerificationTest[(* 35 *)
	OpenVDBLink`OpenVDBMember[bmr, {0, 0, 0}->"Index"]
	,
	1	
]

VerificationTest[(* 36 *)
	OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{1, 0, 0}	
]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{1, 0, 0}	
]

VerificationTest[(* 38 *)
	OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{1, 1, 0}	
]

VerificationTest[(* 39 *)
	OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{1, 0, 0}	
]

VerificationTest[(* 40 *)
	OpenVDBLink`OpenVDBMember[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{1, 1, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBNearest"]

VerificationTest[(* 41 *)
	OpenVDBLink`OpenVDBNearest[bmr, {0, 0, 0}]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 42 *)
	OpenVDBLink`OpenVDBNearest[bmr, {0, 0, 0}->"World"]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 43 *)
	OpenVDBLink`OpenVDBNearest[bmr, {0, 0, 0}->"Index"]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 44 *)
	OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.949999988079071, 0.6439507007598877, 0.8500000238418579}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 45 *)
	OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.949999988079071, 0.6439507007598877, 0.8500000238418579}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 46 *)
	OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.057554371654987335, 0.3408862352371216, -0.37366983294487}, {0.75, 0.8568175435066223, 0.05000000074505806}}	
]

VerificationTest[(* 47 *)
	OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {-1.4500000476837158, 1.2451621294021606, -1.5499999523162842}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 48 *)
	OpenVDBLink`OpenVDBNearest[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{{-0.11179886013269424, 0.3025950789451599, -0.6013116836547852}, {-1.5499999523162842, 1.4629615545272827, -1.649999976158142}, {1.9250102043151855, 1.1039071083068848, 0.6890893578529358}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDistance"]

VerificationTest[(* 49 *)
	OpenVDBLink`OpenVDBDistance[bmr, {0, 0, 0}]
	,
	0.4796196520328522	
]

VerificationTest[(* 50 *)
	OpenVDBLink`OpenVDBDistance[bmr, {0, 0, 0}->"World"]
	,
	0.4796196520328522	
]

VerificationTest[(* 51 *)
	OpenVDBLink`OpenVDBDistance[bmr, {0, 0, 0}->"Index"]
	,
	0.4796196520328522	
]

VerificationTest[(* 52 *)
	OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 53 *)
	OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 54 *)
	OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{0.4796196520328522, 0.5330955982208252, 1.171267032623291}	
]

VerificationTest[(* 55 *)
	OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{0.4796196520328522, 0.08216683566570282, 19.418411254882812}	
]

VerificationTest[(* 56 *)
	OpenVDBLink`OpenVDBDistance[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{0.6823770999908447, 0.18401333689689636, 19.157054901123047}	
]

EndTestSection[]

BeginTestSection["OpenVDBSignedDistance"]

VerificationTest[(* 57 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {0, 0, 0}]
	,
	-0.4796196520328522	
]

VerificationTest[(* 58 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {0, 0, 0}->"World"]
	,
	-0.4796196520328522	
]

VerificationTest[(* 59 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {0, 0, 0}->"Index"]
	,
	-0.4796196520328522	
]

VerificationTest[(* 60 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{-0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 61 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{-0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 62 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{-0.4796196520328522, -0.5330955982208252, 1.171267032623291}	
]

VerificationTest[(* 63 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{-0.4796196520328522, 0.08216683566570282, 19.418411254882812}	
]

VerificationTest[(* 64 *)
	OpenVDBLink`OpenVDBSignedDistance[bmr, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{-0.6823770999908447, -0.18401333689689636, 19.157054901123047}	
]

EndTestSection[]

BeginTestSection["OpenVDBFillWithBalls"]

VerificationTest[(* 65 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.850422203540802], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.823486864566803], Ball[{2.009246826171875, -0.06882219016551971, 0.5779034495353699}, 0.4777902662754059], Ball[{-0.12714575231075287, -0.005113322287797928, 1.2606825828552246}, 0.35073110461235046], Ball[{-1.526516079902649, -0.712378740310669, -0.3921402394771576}, 0.28030794858932495], Ball[{-2.1453604698181152, 0.010189628228545189, 0.384989470243454}, 0.2690991461277008], Ball[{-1.5757085084915161, 0.7037442922592163, -0.4648127257823944}, 0.2654295563697815], Ball[{2.6043074131011963, 0.013343471102416515, 0.15157577395439148}, 0.2584797739982605], Ball[{0.47152549028396606, -0.5848805904388428, -0.5310792922973633}, 0.25168734788894653], Ball[{1.6682215929031372, 0.3148799538612366, 1.0882660150527954}, 0.24282312393188477]}	
]

VerificationTest[(* 66 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, 0.4]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 67 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, 0.4->"World"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 68 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, 5->"Index"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.5], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.5], Ball[{1.6810476779937744, 0.02073744684457779, 0.6675142645835876}, 0.5], Ball[{-0.1838451623916626, -0.08930399268865585, 0.9962171316146851}, 0.5], Ball[{-0.18146087229251862, 0.3106812536716461, 0.1669459491968155}, 0.4175274968147278], Ball[{-1.919439673423767, 0.025230342522263527, 0.49606984853744507}, 0.3911125361919403], Ball[{-1.051155924797058, -0.011858096346259117, 1.3127870559692383}, 0.3438103497028351], Ball[{-1.332594871520996, -0.594613254070282, -0.12945058941841125}, 0.3405416011810303], Ball[{-1.4321969747543335, 0.5771156549453735, -0.047908470034599304}, 0.3383449912071228], Ball[{0.5890829563140869, -0.5093502402305603, -0.3048432171344757}, 0.33354848623275757]}	
]

VerificationTest[(* 69 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {0.35, 0.45}]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 70 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {0.35, 0.45}->"World"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 71 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {4, 5}->"Index"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.5], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.5], Ball[{1.6810476779937744, 0.02073744684457779, 0.6675142645835876}, 0.5], Ball[{-0.1838451623916626, -0.08930399268865585, 0.9962171316146851}, 0.5], Ball[{-0.18146087229251862, 0.3106812536716461, 0.1669459491968155}, 0.4175274968147278]}	
]

VerificationTest[(* 72 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"PackedArray"]
	,
	{{-0.988649845123291, -0.021086128428578377, 0.4303321838378906, 0.5}, {0.7211140394210815, -0.023859607055783272, 0.31998419761657715, 0.5}, {1.8507946729660034, -0.04496168717741966, 0.617486834526062, 0.5}, {-0.13994592428207397, -0.026915403082966805, 1.1647497415542603, 0.5}, {-2.0286922454833984, -0.03874018415808678, 0.3845032751560211, 0.5}, {-1.3945460319519043, 0.6449244022369385, -0.24553744494915009, 0.49731454253196716}, {-0.15341486036777496, 0.0999574139714241, -0.13156546652317047, 0.4919821619987488}, {-1.4079900979995728, -0.7229345440864563, -0.29437118768692017, 0.4842650592327118}, {0.48455536365509033, -0.6056855320930481, -0.5477679967880249, 0.44518759846687317}, {0.6121625900268555, 0.6205933094024658, -0.36544620990753174, 0.43307414650917053}}	
]

VerificationTest[(* 73 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"PackedArray", "SeedCount"->100]
	,
	{{-0.7547315359115601, 0.05440600961446762, 0.347009539604187, 0.5}, {0.7233193516731262, -0.17930136620998383, 0.14818572998046875, 0.5}, {-1.7896685600280762, 0.27292710542678833, 0.5941722989082336, 0.47644609212875366}, {0.26624396443367004, 0.37983623147010803, 0.9525424242019653, 0.46441492438316345}}	
]

VerificationTest[(* 74 *)
	OpenVDBLink`OpenVDBFillWithBalls[bmr, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"Regions", "SeedCount"->100, "Overlapping"->True]
	,
	{Ball[{-0.7547315359115601, 0.05440600961446762, 0.347009539604187}, 0.5], Ball[{-1.0116149187088013, -0.06442729383707047, 0.7733262181282043}, 0.5], Ball[{0.7233193516731262, -0.17930136620998383, 0.14818572998046875}, 0.5], Ball[{0.42385509610176086, 0.13966605067253113, 0.533294677734375}, 0.5], Ball[{-0.11725815385580063, 0.18565157055854797, 0.32102179527282715}, 0.5], Ball[{-0.3044532537460327, -0.38388770818710327, 0.8187362551689148}, 0.5], Ball[{-1.1105129718780518, -0.5051482915878296, 0.4188476800918579}, 0.5], Ball[{-1.4377473592758179, -0.01816057413816452, 0.007965157739818096}, 0.5], Ball[{-0.950627863407135, -0.24608100950717926, -0.016901224851608276}, 0.5], Ball[{1.1687395572662354, 0.3087864816188812, 0.7391563057899475}, 0.5]}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 75 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];maxcoord=MaximalBy[MeshCoordinates[bmr], #1[[2]]&][[1]];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBMember"]

VerificationTest[(* 76 *)
	OpenVDBLink`OpenVDBMember[vdb, {0, 0, 0}]
	,
	1	
]

VerificationTest[(* 77 *)
	OpenVDBLink`OpenVDBMember[vdb, {0, 0, 0}->"World"]
	,
	1	
]

VerificationTest[(* 78 *)
	OpenVDBLink`OpenVDBMember[vdb, {0, 0, 0}->"Index"]
	,
	1	
]

VerificationTest[(* 79 *)
	OpenVDBLink`OpenVDBMember[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{1, 0, 0}	
]

VerificationTest[(* 80 *)
	OpenVDBLink`OpenVDBMember[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{1, 0, 0}	
]

VerificationTest[(* 81 *)
	OpenVDBLink`OpenVDBMember[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{1, 1, 0}	
]

VerificationTest[(* 82 *)
	OpenVDBLink`OpenVDBMember[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{1, 0, 0}	
]

VerificationTest[(* 83 *)
	OpenVDBLink`OpenVDBMember[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{1, 1, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBNearest"]

VerificationTest[(* 84 *)
	OpenVDBLink`OpenVDBNearest[vdb, {0, 0, 0}]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 85 *)
	OpenVDBLink`OpenVDBNearest[vdb, {0, 0, 0}->"World"]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 86 *)
	OpenVDBLink`OpenVDBNearest[vdb, {0, 0, 0}->"Index"]
	,
	{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}	
]

VerificationTest[(* 87 *)
	OpenVDBLink`OpenVDBNearest[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.949999988079071, 0.6439507007598877, 0.8500000238418579}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 88 *)
	OpenVDBLink`OpenVDBNearest[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.949999988079071, 0.6439507007598877, 0.8500000238418579}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 89 *)
	OpenVDBLink`OpenVDBNearest[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {0.05755436420440674, 0.3408862352371216, -0.37366983294487}, {0.75, 0.8568175435066223, 0.05000000074505806}}	
]

VerificationTest[(* 90 *)
	OpenVDBLink`OpenVDBNearest[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{{-0.05000000074505806, 0.2327173799276352, -0.4163863956928253}, {-1.4500000476837158, 1.2451621294021606, -1.5499999523162842}, {2.0162456035614014, 0.8207287192344666, 0.5721988677978516}}	
]

VerificationTest[(* 91 *)
	OpenVDBLink`OpenVDBNearest[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{{-0.11179891228675842, 0.3025951087474823, -0.6013116836547852}, {-1.5499999523162842, 1.4629615545272827, -1.649999976158142}, {1.9250102043151855, 1.1039071083068848, 0.6890893578529358}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDistance"]

VerificationTest[(* 92 *)
	OpenVDBLink`OpenVDBDistance[vdb, {0, 0, 0}]
	,
	0.4796196520328522	
]

VerificationTest[(* 93 *)
	OpenVDBLink`OpenVDBDistance[vdb, {0, 0, 0}->"World"]
	,
	0.4796196520328522	
]

VerificationTest[(* 94 *)
	OpenVDBLink`OpenVDBDistance[vdb, {0, 0, 0}->"Index"]
	,
	0.4796196520328522	
]

VerificationTest[(* 95 *)
	OpenVDBLink`OpenVDBDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 96 *)
	OpenVDBLink`OpenVDBDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 97 *)
	OpenVDBLink`OpenVDBDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{0.4796196520328522, 0.5330955982208252, 1.171267032623291}	
]

VerificationTest[(* 98 *)
	OpenVDBLink`OpenVDBDistance[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{0.4796196520328522, 0.08216683566570282, 19.418411254882812}	
]

VerificationTest[(* 99 *)
	OpenVDBLink`OpenVDBDistance[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{0.6823771595954895, 0.18401333689689636, 19.157054901123047}	
]

EndTestSection[]

BeginTestSection["OpenVDBSignedDistance"]

VerificationTest[(* 100 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {0, 0, 0}]
	,
	-0.4796196520328522	
]

VerificationTest[(* 101 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {0, 0, 0}->"World"]
	,
	-0.4796196520328522	
]

VerificationTest[(* 102 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {0, 0, 0}->"Index"]
	,
	-0.4796196520328522	
]

VerificationTest[(* 103 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}]
	,
	{-0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 104 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"World"]
	,
	{-0.4796196520328522, 0.38957810401916504, 19.418411254882812}	
]

VerificationTest[(* 105 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {{0, 0, 0}, {1, 1, 1}, {5, 20, 0}}->"Index"]
	,
	{-0.4796196520328522, -0.5330955982208252, 1.171267032623291}	
]

VerificationTest[(* 106 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.]
	,
	{-0.4796196520328522, 0.08216683566570282, 19.418411254882812}	
]

VerificationTest[(* 107 *)
	OpenVDBLink`OpenVDBSignedDistance[vdb, {{0, 0, 0}, maxcoord, {5, 20, 0}}->"World", "IsoValue"->0.2]
	,
	{-0.6823771595954895, -0.18401333689689636, 19.157054901123047}	
]

EndTestSection[]

BeginTestSection["OpenVDBFillWithBalls"]

VerificationTest[(* 108 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.850422203540802], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.823486864566803], Ball[{2.009246826171875, -0.06882219016551971, 0.5779034495353699}, 0.4777902662754059], Ball[{-0.12714575231075287, -0.005113322287797928, 1.2606825828552246}, 0.35073110461235046], Ball[{-1.526516079902649, -0.712378740310669, -0.3921402394771576}, 0.28030794858932495], Ball[{-2.1453604698181152, 0.010189628228545189, 0.384989470243454}, 0.2690991461277008], Ball[{-1.5757085084915161, 0.7037442922592163, -0.4648127257823944}, 0.2654295563697815], Ball[{2.6043074131011963, 0.013343471102416515, 0.15157577395439148}, 0.2584797739982605], Ball[{0.47152549028396606, -0.5848805904388428, -0.5310792922973633}, 0.25168734788894653], Ball[{1.6682215929031372, 0.3148799538612366, 1.0882660150527954}, 0.24282312393188477]}	
]

VerificationTest[(* 109 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, 0.4]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 110 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, 0.4->"World"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 111 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, 5->"Index"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.5], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.5], Ball[{1.6810476779937744, 0.02073744684457779, 0.6675142645835876}, 0.5], Ball[{-0.1838451623916626, -0.08930399268865585, 0.9962171316146851}, 0.5], Ball[{-0.18146087229251862, 0.3106812536716461, 0.1669459491968155}, 0.4175274968147278], Ball[{-1.919439673423767, 0.025230342522263527, 0.49606984853744507}, 0.3911125361919403], Ball[{-1.051155924797058, -0.011858096346259117, 1.3127870559692383}, 0.3438103497028351], Ball[{-1.332594871520996, -0.594613254070282, -0.12945058941841125}, 0.3405416011810303], Ball[{-1.4321969747543335, 0.5771156549453735, -0.047908470034599304}, 0.3383449912071228], Ball[{0.5890829563140869, -0.5093502402305603, -0.3048432171344757}, 0.33354848623275757]}	
]

VerificationTest[(* 112 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {0.35, 0.45}]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 113 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {0.35, 0.45}->"World"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.4000000059604645], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.4000000059604645], Ball[{-0.15047386288642883, 0.021544326096773148, 0.9199205636978149}, 0.4000000059604645], Ball[{1.576625943183899, -0.0020190628711134195, 0.6476761698722839}, 0.4000000059604645], Ball[{-0.2173123061656952, 0.0005989930359646678, 0.006176058668643236}, 0.4000000059604645], Ball[{-1.8740158081054688, -0.004608903080224991, 0.5532708168029785}, 0.4000000059604645], Ball[{-1.0420292615890503, -0.00288779498077929, 1.2363476753234863}, 0.4000000059604645], Ball[{-1.3521610498428345, -0.5770297646522522, -0.012899624183773994}, 0.3776744306087494], Ball[{-1.3367924690246582, 0.5725210905075073, 0.055250369012355804}, 0.36751267313957214], Ball[{0.5514872074127197, -0.453774094581604, -0.31019458174705505}, 0.36154019832611084]}	
]

VerificationTest[(* 114 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {4, 5}->"Index"]
	,
	{Ball[{-1.0257866382598877, -0.02364376001060009, 0.42530253529548645}, 0.5], Ball[{0.6637988686561584, 0.0032201975118368864, 0.3482204079627991}, 0.5], Ball[{1.6810476779937744, 0.02073744684457779, 0.6675142645835876}, 0.5], Ball[{-0.1838451623916626, -0.08930399268865585, 0.9962171316146851}, 0.5], Ball[{-0.18146087229251862, 0.3106812536716461, 0.1669459491968155}, 0.4175274968147278]}	
]

VerificationTest[(* 115 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"PackedArray"]
	,
	{{-0.988649845123291, -0.021086128428578377, 0.4303321838378906, 0.5}, {0.7211140394210815, -0.023859607055783272, 0.31998419761657715, 0.5}, {1.8507946729660034, -0.04496168717741966, 0.617486834526062, 0.5}, {-0.13994592428207397, -0.026915403082966805, 1.1647497415542603, 0.5}, {-2.0286922454833984, -0.03874018415808678, 0.3845032751560211, 0.5}, {-1.3945460319519043, 0.6449244022369385, -0.24553744494915009, 0.49731454253196716}, {-0.15341486036777496, 0.0999574139714241, -0.13156546652317047, 0.4919821619987488}, {-1.4079900979995728, -0.7229345440864563, -0.29437118768692017, 0.4842650592327118}, {0.48455536365509033, -0.6056855320930481, -0.5477679967880249, 0.44518759846687317}, {0.6121625900268555, 0.6205933094024658, -0.36544620990753174, 0.43307414650917053}}	
]

VerificationTest[(* 116 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"PackedArray", "SeedCount"->100]
	,
	{{-0.7547315359115601, 0.05440600961446762, 0.347009539604187, 0.5}, {0.7233193516731262, -0.17930136620998383, 0.14818572998046875, 0.5}, {-1.7896685600280762, 0.27292710542678833, 0.5941722989082336, 0.47644609212875366}, {0.26624396443367004, 0.37983623147010803, 0.9525424242019653, 0.46441492438316345}}	
]

VerificationTest[(* 117 *)
	OpenVDBLink`OpenVDBFillWithBalls[vdb, 10, {4, 5}->"Index", "IsoValue"->0.2, "ReturnType"->"Regions", "SeedCount"->100, "Overlapping"->True]
	,
	{Ball[{-0.7547315359115601, 0.05440600961446762, 0.347009539604187}, 0.5], Ball[{-1.0116149187088013, -0.06442729383707047, 0.7733262181282043}, 0.5], Ball[{0.7233193516731262, -0.17930136620998383, 0.14818572998046875}, 0.5], Ball[{0.42385509610176086, 0.13966605067253113, 0.533294677734375}, 0.5], Ball[{-0.11725815385580063, 0.18565157055854797, 0.32102179527282715}, 0.5], Ball[{-0.3044532537460327, -0.38388770818710327, 0.8187362551689148}, 0.5], Ball[{-1.1105129718780518, -0.5051482915878296, 0.4188476800918579}, 0.5], Ball[{-1.4377473592758179, -0.01816057413816452, 0.007965157739818096}, 0.5], Ball[{-0.950627863407135, -0.24608100950717926, -0.016901224851608276}, 0.5], Ball[{1.1687395572662354, 0.3087864816188812, 0.7391563057899475}, 0.5]}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
