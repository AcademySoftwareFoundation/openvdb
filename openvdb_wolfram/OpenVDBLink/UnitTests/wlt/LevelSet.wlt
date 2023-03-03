BeginTestSection["Level Set Creation Tests"]

BeginTestSection["Generic"]

BeginTestSection["OpenVDBLevelSet"]

VerificationTest[(* 1 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBLevelSet]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 2 *)
	Attributes[OpenVDBLink`OpenVDBLevelSet]
	,
	{Protected,  ReadProtected}	
]

VerificationTest[(* 3 *)
	Options[OpenVDBLink`OpenVDBLevelSet]
	,
	{"Creator" :> OpenVDBLink`$OpenVDBCreator,  "Name" -> None, "ScalarType" -> "Float"}	
]

VerificationTest[(* 4 *)
	SyntaxInformation[OpenVDBLink`OpenVDBLevelSet]
	,
	{"ArgumentsPattern" -> {_,  _., _., OptionsPattern[]}}	
]

VerificationTest[(* 5 *)
	{OpenVDBLink`OpenVDBLevelSet[],  OpenVDBLink`OpenVDBLevelSet["error"], 
  OpenVDBLink`OpenVDBLevelSet[Ball[],  "error"], OpenVDBLink`OpenVDBLevelSet[Ball[],  
   0.1, "error"], OpenVDBLink`OpenVDBLevelSet[Ball[],  0.1, 3., "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBLevelSet::argb, OpenVDBLevelSet::reg, OpenVDBLevelSet::nonpos, OpenVDBLevelSet::nonpos, OpenVDBLevelSet::nonopt}
]

VerificationTest[(* 6 *)
	(OpenVDBLink`OpenVDBLevelSet[OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ 
  {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", 
   "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, 
  $Failed, $Failed}
	,
	{OpenVDBLevelSet::reg, OpenVDBLevelSet::reg, OpenVDBLevelSet::reg, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["OpenVDBLevelSet"]

VerificationTest[(* 7 *)
	OpenVDBLink`OpenVDBLevelSet[Ball[],  0.1, 3.]["ActiveVoxelCount"]
	,
	7674	
]

VerificationTest[(* 8 *)
	OpenVDBLink`$OpenVDBHalfWidth = 3.; OpenVDBLink`OpenVDBLevelSet[Ball[],  0.1][
   "ActiveVoxelCount"]
	,
	7674	
]

VerificationTest[(* 9 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`OpenVDBLevelSet[Ball[]][
   "ActiveVoxelCount"]
	,
	7674	
]

VerificationTest[(* 10 *)
	specialRegions = {EmptyRegion[3],  Cube[], Dodecahedron[], Icosahedron[], 
    Octahedron[], Tetrahedron[], Hexahedron[], Pyramid[], Prism[], Cuboid[], 
    Parallelepiped[], Simplex[3], Sphere[], Ball[], Ellipsoid[{0,  0, 0},  {1,  2, 3}], 
    Cylinder[], Cone[], CapsuleShape[], Tube[{{0,  0, 0},  {1,  0, 0}},  0.3], 
    SphericalShell[], Torus[], FilledTorus[], 
    Polyhedron[{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {{2,  3, 4},  {3,  2, 1}, {4,  1, 2}, {1,  4, 3}}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ specialRegions
	,
	{0,  3086, 12448, 5382, 2426, 1254, 3086, 5220, 2397, 3086, 4532, 1444, 7674, 7674, 
  29581, 10638, 5834, 14743, 2015, 9060, 4275, 4275, 1444}	
]

VerificationTest[(* 11 *)
	multiSetRegions = {Ball[{{0,  0, 0},  {2,  0, 0}},  1],  
    Hexahedron[(#1 + {{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}, {0,  0, 1}, 
         {1,  0, 1}, {1,  1, 1}, {0,  1, 1}} & ) /@ {0,  3}], 
    Cone[(#1 + {{0,  0, -1},  {0,  0, 1}} & ) /@ {0,  3}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ multiSetRegions
	,
	{14699,  6172, 11668}	
]

VerificationTest[(* 12 *)
	formulaRegions = {ImplicitRegion[x^6 - 5*x^4*y + 3*x^4*y^2 + 10*x^2*y^3 + 3*x^2*y^4 - 
       y^5 + y^6 + z^2 <= 1,  {x,  y, z}],  ParametricRegion[
     {{x,  y, z + y*x},  x^2 + y^2 + z^2 <= 1},  {x,  y, z}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ formulaRegions
	,
	{11638,  8035}	
]

VerificationTest[(* 13 *)
	meshRegions = (ExampleData[{"Geometry3D",  "Triceratops"},  #1] & ) /@ 
    {"MeshRegion",  "BoundaryMeshRegion"}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ meshRegions
	,
	{26073,  26073}	
]

VerificationTest[(* 14 *)
	derivedRegions = {RegionUnion[Ball[],  Cuboid[]],  RegionIntersection[Ball[],  
     Cuboid[]], RegionDifference[Ball[],  Cuboid[]], RegionSymmetricDifference[Ball[],  
     Cuboid[]], RegionProduct[ImplicitRegion[x^6 - 5*x^4*y + 3*x^4*y^2 + 10*x^2*y^3 + 
        3*x^2*y^4 - y^5 + y^6 <= 1/100,  {x,  y}],  Line[{{0},  {1}}]]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] > 0 & ) /@ derivedRegions
	,
	{True,  True, True, True, True}	
]

VerificationTest[(* 15 *)
	meshComplexes = {{{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}},  
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     Polygon[{{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}]}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {Polygon[{{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}]}}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  {Polygon[{1,  2, 3}],  
      Polygon[{2,  3, 4}], Polygon[{1,  3, 4}], Polygon[{1,  2, 4}]}}}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ meshComplexes
	,
	{1444,  1444, 1444, 1444}	
]

VerificationTest[(* 16 *)
	tubeComplexes = {{MeshRegion[{{0,  0, 0},  {1,  0, 0}},  Line[{1,  2}]],  0.3},  
    {{{0,  0, 0},  {1,  0, 0}},  {1,  2}, 0.3}, {{{0,  0, 0},  {1,  0, 0}},  {{1,  2}}, 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  {{1,  2},  {2,  3}}, 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  Line[{{1,  2},  {2,  3}}], 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  {Line[{1,  2}],  Line[{2,  3}]}, 0.3}}; 
  (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ tubeComplexes
	,
	{2015,  2015, 2015, 3201, 3201, 3201}	
]

VerificationTest[(* 17 *)
	thickMeshComplexes = {{MeshRegion[{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  
      Polygon[{{1,  2, 3},  {1,  3, 4}}]],  0.3},  
    {{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  Polygon[{{1,  2, 3},  {1,  3, 4}}], 
     0.3}, {{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  {{1,  2, 3},  {1,  3, 4}}, 
     0.3}}; (OpenVDBLink`OpenVDBLevelSet[#1]["ActiveVoxelCount"] & ) /@ 
   thickMeshComplexes
	,
	{4175,  4175, 4175}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["OpenVDBLevelSet"]

VerificationTest[(* 18 *)
	OpenVDBLink`OpenVDBLevelSet[Ball[],  0.1, 3., "ScalarType" -> "Double"][
  "ActiveVoxelCount"]
	,
	7728	
]

VerificationTest[(* 19 *)
	OpenVDBLink`$OpenVDBHalfWidth = 3.; OpenVDBLink`OpenVDBLevelSet[Ball[],  0.1, 
    "ScalarType" -> "Double"]["ActiveVoxelCount"]
	,
	7728	
]

VerificationTest[(* 20 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`OpenVDBLevelSet[Ball[],  
    "ScalarType" -> "Double"]["ActiveVoxelCount"]
	,
	7728	
]

VerificationTest[(* 21 *)
	specialRegions = {EmptyRegion[3],  Cube[], Dodecahedron[], Icosahedron[], 
    Octahedron[], Tetrahedron[], Hexahedron[], Pyramid[], Prism[], Cuboid[], 
    Parallelepiped[], Simplex[3], Sphere[], Ball[], Ellipsoid[{0,  0, 0},  {1,  2, 3}], 
    Cylinder[], Cone[], CapsuleShape[], Tube[{{0,  0, 0},  {1,  0, 0}},  0.3], 
    SphericalShell[], Torus[], FilledTorus[], 
    Polyhedron[{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {{2,  3, 4},  {3,  2, 1}, {4,  1, 2}, {1,  4, 3}}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ specialRegions
	,
	{0,  3086, 12448, 5382, 2426, 1254, 3086, 5220, 2397, 3086, 4532, 1444, 7728, 7728, 
  29581, 10638, 5834, 14743, 2017, 9066, 4275, 4275, 1444}	
]

VerificationTest[(* 22 *)
	multiSetRegions = {Ball[{{0,  0, 0},  {2,  0, 0}},  1],  
    Hexahedron[(#1 + {{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}, {0,  0, 1}, 
         {1,  0, 1}, {1,  1, 1}, {0,  1, 1}} & ) /@ {0,  3}], 
    Cone[(#1 + {{0,  0, -1},  {0,  0, 1}} & ) /@ {0,  3}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ multiSetRegions
	,
	{14798,  6172, 11668}	
]

VerificationTest[(* 23 *)
	formulaRegions = {ImplicitRegion[x^6 - 5*x^4*y + 3*x^4*y^2 + 10*x^2*y^3 + 3*x^2*y^4 - 
       y^5 + y^6 + z^2 <= 1,  {x,  y, z}],  ParametricRegion[
     {{x,  y, z + y*x},  x^2 + y^2 + z^2 <= 1},  {x,  y, z}]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ formulaRegions
	,
	{11638,  8035}	
]

VerificationTest[(* 24 *)
	meshRegions = (ExampleData[{"Geometry3D",  "Triceratops"},  #1] & ) /@ 
    {"MeshRegion",  "BoundaryMeshRegion"}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ meshRegions
	,
	{26073,  26073}	
]

VerificationTest[(* 25 *)
	derivedRegions = {RegionUnion[Ball[],  Cuboid[]],  RegionIntersection[Ball[],  
     Cuboid[]], RegionDifference[Ball[],  Cuboid[]], RegionSymmetricDifference[Ball[],  
     Cuboid[]], RegionProduct[ImplicitRegion[x^6 - 5*x^4*y + 3*x^4*y^2 + 10*x^2*y^3 + 
        3*x^2*y^4 - y^5 + y^6 <= 1/100,  {x,  y}],  Line[{{0},  {1}}]]}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"]["ActiveVoxelCount"] > 
     0 & ) /@ derivedRegions
	,
	{True,  True, True, True, True}	
]

VerificationTest[(* 26 *)
	meshComplexes = {{{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}},  
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     Polygon[{{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}]}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  
     {Polygon[{{1,  2, 3},  {2,  3, 4}, {1,  3, 4}, {1,  2, 4}}]}}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}, {0,  0, 1}},  {Polygon[{1,  2, 3}],  
      Polygon[{2,  3, 4}], Polygon[{1,  3, 4}], Polygon[{1,  2, 4}]}}}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ meshComplexes
	,
	{1444,  1444, 1444, 1444}	
]

VerificationTest[(* 27 *)
	tubeComplexes = {{MeshRegion[{{0,  0, 0},  {1,  0, 0}},  Line[{1,  2}]],  0.3},  
    {{{0,  0, 0},  {1,  0, 0}},  {1,  2}, 0.3}, {{{0,  0, 0},  {1,  0, 0}},  {{1,  2}}, 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  {{1,  2},  {2,  3}}, 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  Line[{{1,  2},  {2,  3}}], 0.3}, 
    {{{0,  0, 0},  {1,  0, 0}, {0,  1, 0}},  {Line[{1,  2}],  Line[{2,  3}]}, 0.3}}; 
  (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ tubeComplexes
	,
	{2017,  2017, 2017, 3203, 3203, 3203}	
]

VerificationTest[(* 28 *)
	thickMeshComplexes = {{MeshRegion[{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  
      Polygon[{{1,  2, 3},  {1,  3, 4}}]],  0.3},  
    {{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  Polygon[{{1,  2, 3},  {1,  3, 4}}], 
     0.3}, {{{0,  0, 0},  {1,  0, 0}, {1,  1, 0}, {0,  1, 0}},  {{1,  2, 3},  {1,  3, 4}}, 
     0.3}}; (OpenVDBLink`OpenVDBLevelSet[#1,  "ScalarType" -> "Double"][
     "ActiveVoxelCount"] & ) /@ thickMeshComplexes
	,
	{4175,  4175, 4175}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
