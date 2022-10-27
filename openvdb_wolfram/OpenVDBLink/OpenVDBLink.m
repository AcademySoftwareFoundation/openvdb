(* ::Package:: *)

(* ::Title:: *)
(*OpenVDBLink*)


(* ::Subtitle:: *)
(*Greg Hurst, United Therapeutics*)


(* ::Subsubtitle:: *)
(*ghurst588@gmail.com*)


(* ::Subsubtitle:: *)
(*2021 - 2022*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageImport["OpenVDBLink`LTemplate`"]


PackageExport["$OpenVDBLibrary"]
PackageExport["$OpenVDBInstallationDirectory"]


PackageExport["OpenVDBDocumentation"]


$OpenVDBLibrary::usage = "$OpenVDBLibrary is the full path to the OpenVDB Library loaded by OpenVDBLink.";
$OpenVDBInstallationDirectory::usage = "$OpenVDBInstallationDirectory gives the top-level directory in which your OpenVDB installation resides.";


OpenVDBDocumentation::usage = "OpenVDBDocumentation[] opens the OpenVDBLink documentation.\nOpenVDBDocumentation[\"Web\"] opens the OpenVDBLink in a web browser.";


OpenVDBLink`Developer`Recompile::usage = "OpenVDBLink`Developer`Recompile[] recompiles the OpenVDB library and reloads the functions.";


(* ::Section:: *)
(*Path Variables*)


(* ::Subsection::Closed:: *)
(*Paths*)


$packageDirectory = DirectoryName[$InputFileName];
$systemID = $SystemID;

$libraryDirectory  = FileNameJoin[{$packageDirectory, "LibraryResources", $systemID}];
$sourceDirectory   = FileNameJoin[{$packageDirectory, "Source", "ExplicitGrids"}];
$unitTestDirectory = FileNameJoin[{$packageDirectory, "UnitTests"}];
$buildSettingsFile = FileNameJoin[{$packageDirectory, "BuildSettings.m"}];


(* ::Subsection::Closed:: *)
(*Load build settings*)


Get[$buildSettingsFile];
$LibraryPath = DeleteDuplicates[Prepend[$LibraryPath, $libraryDirectory]];


(* ::Subsection::Closed:: *)
(*$OpenVDBLibrary & $OpenVDBInstallationDirectory*)


$OpenVDBLibrary = FileNameJoin[{$libraryDirectory, $libraryName}];
$OpenVDBInstallationDirectory = $packageDirectory;


(* ::Section:: *)
(*OpenVDB library function template*)


(* ::Subsection::Closed:: *)
(*Class data declarations*)


PackageScope["$scalarType"]
PackageScope["$integerType"]
PackageScope["$vectorType"]
PackageScope["$booleanType"]
PackageScope["$maskType"]
PackageScope["$classTypeList"]
PackageScope["$GridClassData"]


$scalarType  = "Scalar";
$integerType = "Integer";
$vectorType  = "Vector";
$booleanType = "Boolean";
$maskType    = "Mask";


$classTypeList = {$scalarType, $vectorType, $integerType, $booleanType, $maskType};


$GridClassData = <|
    $scalarType  -> <|
        "Double" -> <|"ClassName" -> "OpenVDBDoubleGrid", "TreeName" -> "Tree_double_5_4_3", "WLBaseType" -> Real, "PixelClass" -> True|>,
        "Float"  -> <|"ClassName" -> "OpenVDBFloatGrid",  "TreeName" -> "Tree_float_5_4_3",  "WLBaseType" -> Real, "PixelClass" -> True, "Alias" -> "Scalar"|>
    |>,

    $integerType -> <|
        "Byte"   -> <|"ClassName" -> "OpenVDBByteGrid",   "TreeName" -> "Tree_uint8_5_4_3",  "WLBaseType" -> Integer, "PixelClass" -> True|>,
        "Int32"  -> <|"ClassName" -> "OpenVDBInt32Grid",  "TreeName" -> "Tree_int32_5_4_3",  "WLBaseType" -> Integer|>,
        "Int64"  -> <|"ClassName" -> "OpenVDBInt64Grid",  "TreeName" -> "Tree_int64_5_4_3",  "WLBaseType" -> Integer|>,
        "UInt32" -> <|"ClassName" -> "OpenVDBUInt32Grid", "TreeName" -> "Tree_uint32_5_4_3", "WLBaseType" -> Integer|>
    |>,

    $vectorType -> <|
        "Vec2D" -> <|"ClassName" -> "OpenVDBVec2DGrid", "TreeName" -> "Tree_vec2d_5_4_3", "WLBaseType" -> Real|>,
        "Vec2I" -> <|"ClassName" -> "OpenVDBVec2IGrid", "TreeName" -> "Tree_vec2i_5_4_3", "WLBaseType" -> Integer|>,
        "Vec2S" -> <|"ClassName" -> "OpenVDBVec2SGrid", "TreeName" -> "Tree_vec2s_5_4_3", "WLBaseType" -> Real|>,
        "Vec3D" -> <|"ClassName" -> "OpenVDBVec3DGrid", "TreeName" -> "Tree_vec3d_5_4_3", "WLBaseType" -> Real|>,
        "Vec3I" -> <|"ClassName" -> "OpenVDBVec3IGrid", "TreeName" -> "Tree_vec3i_5_4_3", "WLBaseType" -> Integer|>,
        "Vec3S" -> <|"ClassName" -> "OpenVDBVec3SGrid", "TreeName" -> "Tree_vec3s_5_4_3", "WLBaseType" -> Real, "Alias" -> "Vector"|>
    |>,

    $booleanType -> <|
        $booleanType -> <|"ClassName" -> "OpenVDBBoolGrid", "TreeName" -> "Tree_bool_5_4_3", "WLBaseType" -> Integer, "PixelClass" -> True|>
    |>,

    $maskType -> <|
        $maskType -> <|"ClassName" -> "OpenVDBMaskGrid", "TreeName" -> "Tree_mask_5_4_3", "PixelClass" -> True|>
    |>
|>;


$scalarClass = "OpenVDBFloatGrid";
$colorVectorClass = "OpenVDBVec3SGrid";


Scan[(pixelClassQ[#["ClassName"]] = TrueQ[#["PixelClass"]])&, Values[Join @@ $GridClassData]];


KeyValueMap[(pixelTypeQ[#1] = TrueQ[#2["PixelClass"]])&, Join @@ $GridClassData];


(* ::Subsection::Closed:: *)
(*Types*)


(* ::Subsubsection::Closed:: *)
(*Types*)


PackageScope["$gridTypeList"]
PackageScope["$internalTypeList"]
PackageScope["fromInternalType"]
PackageScope["typeClass"]
PackageScope["aliasTypeQ"]
PackageScope["resolveAliasType"]


Block[{alltypes, typetreegroups},
    alltypes = Join @@ Values[$GridClassData];

    typetreegroups = Join[
        {#["Alias"], #["TreeName"], #["ClassName"]}& /@ Values[alltypes],
        KeyValueMap[{#1, #2["TreeName"], #2["ClassName"]}&, alltypes]
    ];

    typetreegroups = Transpose[DeleteMissing[typetreegroups, 1, 2]];

    $gridTypeList = typetreegroups[[1]];
    $internalTypeList = typetreegroups[[2]];

    Clear[typeClass];
    Block[{$Context = "OpenVDBLink`LTemplate`Classes`"},
        MapThread[(typeClass[#1] = Symbol[#3])&, typetreegroups];
    ];

    KeyValueMap[Function[{k, v},
        If[!MissingQ[v["Alias"]],
            aliasTypeQ[v["Alias"]] = True;
            resolveAliasType[v["Alias"]] = k;
        ]
    ], alltypes];

    aliasTypeQ[_] = False;
]


toInternalType = AssociationThread[$gridTypeList, $internalTypeList];
fromInternalType = AssociationThread[$internalTypeList, $gridTypeList];


(* ::Subsubsection::Closed:: *)
(*typeGridName*)


PackageScope["typeGridName"]


Clear[typeGridName];


KeyValueMap[Function[{k, v}, typeGridName[k] = v["ClassName"]], Join @@ Values[$GridClassData]];


(* ::Subsubsection::Closed:: *)
(*gridType*)


PackageScope["gridType"]


Clear[gridType];


KeyValueMap[Function[{k, v}, gridType[v["ClassName"]] = k], Join @@ Values[$GridClassData]];


(* ::Subsection::Closed:: *)
(*$OpenVDBTemplate*)


(* ::Subsubsection::Closed:: *)
(*BaseGrid template*)


LVDBBaseGridClass[class_String, members_List] := LClass[class, Join[vdbBaseGridMembers[class], vdbBaseGridPixelMembers[class], members]]


vdbBaseGridMembers[class_] :=
    {
        (* ------------ creation, deletion ------------ *)
        LFun["createEmptyGrid", {}, "Void"],
        LFun["deleteGrid", {}, "Void"],
        LFun["copyGrid", {LExpressionID[class]}, "Void"],

        (* ------------ IO ------------ *)
        LFun["importVDBType", {"UTF8String", "UTF8String"}, "UTF8String"],
        LFun["importVDB", {"UTF8String", "UTF8String"}, "Boolean"],
        LFun["exportVDB", {"UTF8String"}, "Void"],

        (* ------------ setters ------------ *)
        LFun["setActiveStates", {{Integer, 2, "Constant"}, {Integer, 1, "Constant"}}, "Void"],
        LFun["setGridClass", {Integer}, "Void"],
        LFun["setCreator", {"UTF8String"}, "Void"],
        LFun["setName", {"UTF8String"}, "Void"],
        LFun["setVoxelSize", {Real}, "Void"],

        (* ------------ getters ------------ *)
        LFun["getActiveStates", {{Integer, 2, "Constant"}}, {Integer, 1}],
        LFun["getActiveLeafVoxelCount", {}, Integer],
        LFun["getActiveTileCount", {}, Integer],
        LFun["getActiveVoxelCount", {}, Integer],
        LFun["getGridClass", {}, Integer],
        LFun["getCreator", {}, "UTF8String"],
        LFun["getGridBoundingBox", {}, {Integer, 2}],
        LFun["getGridDimensions", {}, {Integer, 1}],
        LFun["getGridType", {}, "UTF8String"],
        LFun["getHasUniformVoxels", {}, "Boolean"],
        LFun["getIsEmpty", {}, "Boolean"],
        LFun["getMemoryUsage", {}, Integer],
        LFun["getName", {}, "UTF8String"],
        LFun["getVoxelSize", {}, Real],

        (* ------------ CSG ------------ *)
        LFun["gridMax", {LExpressionID[class]}, "Void"],
        LFun["gridMin", {LExpressionID[class]}, "Void"],

        (* ------------ Metadata ------------ *)
        LFun["getBooleanMetadata", {"UTF8String"}, "Boolean"],
        LFun["getIntegerMetadata", {"UTF8String"}, Integer],
        LFun["getRealMetadata", {"UTF8String"}, Real],
        LFun["getStringMetadata", {"UTF8String"}, "UTF8String"],
        LFun["setBooleanMetadata", {"UTF8String", "Boolean"}, "Void"],
        LFun["setStringMetadata", {"UTF8String", "UTF8String"}, "Void"],
        LFun["setDescription", {"UTF8String"}, "Void"],

        (* ------------ grid transformation ------------ *)
        LFun["transformGrid", {LExpressionID[class], {Real, 2, "Constant"}, Integer}, "Void"],

        (* ------------ aggregate data ------------ *)
        LFun["sliceVoxelCounts", {Integer, Integer}, {Integer, 1}],
        LFun["activeTiles", {{Integer, 2, "Constant"}, "Boolean"}, {Integer, 3}],
        LFun["activeVoxelPositions", {{Integer, 2, "Constant"}}, {Integer, 2}]
    };


vdbBaseGridPixelMembers[class_?pixelClassQ] :=
{
    (* ------------ Image ------------ *)
    LFun["depthMap", {{Integer, 2, "Constant"}, Real, Real, Real}, LType["Image", "Real32"]],
    LFun["gridSliceImage", {Integer, {Integer, 2, "Constant"}, "Boolean", "Boolean"}, LType["Image"]],
    LFun["gridImage3D", {{Integer, 2, "Constant"}}, LType["Image3D"]]
}


vdbBaseGridPixelMembers[___] = {};


(* ::Subsubsection::Closed:: *)
(*BaseNumericGrid template*)


LVDBBaseNumericGrid[class_String, type_, rank_, members_List] :=
    LClass[
        class,
        Join[
            vdbBaseGridMembers[class],
            vdbBaseGridPixelMembers[class],
            vdbBaseNumericGridMembers[class, type, rank],
            members
        ]
    ]


vdbBaseNumericGridMembers[class_, type_, rank_] :=
With[{
    $scalarinput = scalarInput[type, rank], $scalaroutput = scalarOutput[type, rank],
    $vectorinput = vectorInput[type, rank], $vectoroutput = vectorOutput[type, rank],
    $matrixinput = matrixInput[type, rank], $matrixoutput = matrixOutput[type, rank],
    $cubeinput   = cubeInput[type, rank],   $cubeoutput   = cubeOutput[type, rank]
},
    {
        (* ------------ setters ------------ *)
        LFun["setBackgroundValue", {$scalarinput}, "Void"],
        LFun["setValues", {{Integer, 2, "Constant"}, $vectorinput}, "Void"],

        (* ------------ getters ------------ *)
        LFun["getBackgroundValue", {}, $scalaroutput],
        LFun["getMinMaxValues", {}, $vectoroutput],
        LFun["getValues", {{Integer, 2, "Constant"}}, $vectoroutput],

        (* ------------ aggregate data ------------ *)
        LFun["sliceVoxelValueTotals", {Integer, Integer}, $vectoroutput],
        LFun["activeVoxelValues", {{Integer, 2, "Constant"}}, $vectoroutput],
        LFun["gridSlice", {Integer, {Integer, 2, "Constant"}, "Boolean", "Boolean"}, $matrixoutput],
        LFun["gridData", {{Integer, 2, "Constant"}}, $cubeoutput]
    }
];


scalarInput[type_, 0]     := type
scalarInput[type_, rank_] := {type, rank, "Constant"}
vectorInput[type_, rank_] := {type, rank+1, "Constant"}
matrixInput[type_, rank_] := {type, rank+2, "Constant"}
cubeInput[type_, rank_]   := {type, rank+3, "Constant"}


scalarOutput[type_, 0]     := type
scalarOutput[type_, rank_] := {type, rank}
vectorOutput[type_, rank_] := {type, rank+1}
matrixOutput[type_, rank_] := {type, rank+2}
cubeOutput[type_, rank_]   := {type, rank+3}


(* ::Subsubsection::Closed:: *)
(*ScalarGrid template*)


LVDBScalarGridClass[class_, type_] :=
    LVDBBaseNumericGrid[class, type, 0,
        {
            (* ------------ getters ------------ *)
            LFun["getHalfwidth", {}, Real],

            (* ------------ CSG ------------ *)
            LFun["gridUnion", {LExpressionID[class]}, "Void"],
            LFun["gridIntersection", {LExpressionID[class]}, "Void"],
            LFun["gridDifference", {LExpressionID[class]}, "Void"],
            LFun["gridUnionCopy", {{Integer, 1, "Constant"}}, "Void"],
            LFun["gridIntersectionCopy", {{Integer, 1, "Constant"}}, "Void"],
            LFun["gridDifferenceCopy", {LExpressionID[class], LExpressionID[class]}, "Void"],
            LFun["clipGrid", {LExpressionID[class], {Real, 2, "Constant"}}, "Void"],

            (* ------------ level set creation ------------ *)
            LFun["ballLevelSet", {{Real, 1, "Constant"}, Real, Real, Real, "Boolean"}, "Void"],
            LFun["cuboidLevelSet", {{Real, 2, "Constant"}, Real, Real, "Boolean"}, "Void"],
            LFun["meshLevelSet", {{Real, 2, "Constant"}, {Integer, 2, "Constant"}, Real, Real, "Boolean"}, "Void"],
            LFun["offsetSurfaceLevelSet", {{Real, 2, "Constant"}, {Integer, 2, "Constant"}, Real, Real, Real, "Boolean"}, "Void"],

            (* ------------ level set measure ------------ *)
            LFun["levelSetGridArea", {}, Real],
            LFun["levelSetGridEulerCharacteristic", {}, Integer],
            LFun["levelSetGridGenus", {}, Integer],
            LFun["levelSetGridVolume", {}, Real],

            (* ------------ distance measure ------------ *)
            LFun["gridMember", {{Integer, 2, "Constant"}, Real}, {Integer, 1}],
            LFun["gridNearest", {{Real, 2, "Constant"}, Real}, {Real, 2}],
            LFun["gridDistance", {{Real, 2, "Constant"}, Real}, {Real, 1}],
            LFun["gridSignedDistance", {{Real, 2, "Constant"}, Real}, {Real, 1}],
            LFun["fillWithBalls", {Integer, Integer, "Boolean", Real, Real, Real, Integer}, {Real, 2}],

            (* ------------ filters ------------ *)
            LFun["filterGrid", {Integer, Integer, Integer}, "Void"],

            (* ------------ mesh creation ------------ *)
            LFun["meshCellCount", {Real, Real, "Boolean"}, {Integer, 1}],
            LFun["meshData", {Real, Real, "Boolean"}, {Real, 1}],

            (* ------------ fog volume ------------ *)
            LFun["levelSetToFogVolume", {Real}, "Void"],

            (* ------------ grid transformation ------------ *)
            LFun["scalarMultiply", {Real}, "Void"],
            LFun["gammaAdjustment", {Real}, "Void"],

            (* ------------ morphology ------------ *)
            LFun["resizeBandwidth", {Real}, "Void"],
            LFun["offsetLevelSet", {Real}, "Void"],

            (* ------------ render ------------ *)
            LFun["renderGrid", {Real, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"},
                {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, Integer, Integer, Integer, {Integer, 1, "Constant"}, Real,
                {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, "Boolean"}, LType["Image", "Byte"]],

            LFun["renderGridPBR", {Real, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"},
                {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, Integer, Integer, {Integer, 1, "Constant"}, Real, "Boolean",
                {Real, 1}, {Real, 1}, {Real, 1},
                Real, Real, Real, Real,
                {Real, 1}, Real, Real, Real,
                Real, Real, Real
            }, LType["Image", "Byte"]],

            (*LFun["renderGridVectorColor", {Real, LExpressionID[$colorVectorClass], LExpressionID[$colorVectorClass], LExpressionID[$colorVectorClass], {Real, 1, "Constant"},
                {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, Integer, Integer, Integer,
                {Integer, 1, "Constant"}, Real, {Real, 1, "Constant"}, {Real, 1, "Constant"}, {Real, 1, "Constant"}, "Boolean"}, LType["Image", "Byte"]],*)

            (* ------------ aggregate data ------------ *)
            LFun["activeVoxels", {{Integer, 2, "Constant"}}, LType[SparseArray, Real, 3]]
        }
    ]


(* ::Subsubsection::Closed:: *)
(*IntegerGrid template*)


LVDBIntegerGridClass[class_, type_] :=
    LVDBBaseNumericGrid[class, type, 0,
        {
            (* ------------ aggregate data ------------ *)
            LFun["activeVoxels", {{Integer, 2, "Constant"}}, LType[SparseArray, Integer, 3]]
        }
    ]


(* ::Subsubsection::Closed:: *)
(*VectorGrid template*)


LVDBVectorGridClass[class_, type_] :=
    LVDBBaseNumericGrid[class, type, 1,
        {

        }
    ]


(* ::Subsubsection::Closed:: *)
(*BoolGrid template*)


$booleanBlackList = {"sliceVoxelValueTotals"};


deleteNonBooleanFuncs[class_] := class /. {LFun[Alternatives @@ $booleanBlackList, ___] -> Nothing}


LVDBBoolGridClass =
With[{
    class = $GridClassData[$booleanType, $booleanType, "ClassName"],
    type = $GridClassData[$booleanType, $booleanType, "WLBaseType"]
},
    deleteNonBooleanFuncs @ LVDBBaseNumericGrid[class, type, 0,
        {
            (* ------------ aggregate data ------------ *)
            LFun["activeVoxels", {{Integer, 2, "Constant"}}, LType[SparseArray, Integer, 3]]
        }
    ]
];


(* ::Subsubsection::Closed:: *)
(*MaskGrid template*)


LVDBMaskGridClass = LVDBBaseGridClass[$GridClassData[$maskType, $maskType, "ClassName"],
    {

    }
];


(* ::Subsubsection::Closed:: *)
(*Main template*)


$OpenVDBTemplate = LTemplate["OpenVDBLink",
    Join[
        LVDBScalarGridClass [#ClassName, #WLBaseType]& /@ Values[$GridClassData[$scalarType]],
        LVDBIntegerGridClass[#ClassName, #WLBaseType]& /@ Values[$GridClassData[$integerType]],
        LVDBVectorGridClass [#ClassName, #WLBaseType]& /@ Values[$GridClassData[$vectorType]],
        {
            LVDBBoolGridClass,
            LVDBMaskGridClass
        }
    ]
];


(* ::Section:: *)
(*Loader & Compiler*)


(* ::Subsection::Closed:: *)
(*Recompile*)


OpenVDBLink`Developer`Recompile::build = "No build settings found. Please check BuildSettings.m.";


OpenVDBLink`Developer`Recompile[printQ_:False] :=
    (
        If[$buildSettings === None,
            Message[OpenVDBLink`Developer`Recompile::build];
            Return[$Failed]
        ];

        If[!DirectoryQ[$libraryDirectory],
            CreateDirectory[$libraryDirectory]
        ];

        SetDirectory[$sourceDirectory];
        CompileTemplate[
            $OpenVDBTemplate,
            {},
            Sequence @@ If[TrueQ[printQ],
                {"ShellCommandFunction" -> Print, "ShellOutputFunction" -> Print},
                {}
            ],
            "CleanIntermediate" -> True,
            "TargetDirectory" -> $libraryDirectory,
            Sequence @@ $buildSettings
        ];
        ResetDirectory[];

        LoadOpenVDBLink[]
    )


(* ::Subsection::Closed:: *)
(*LoadOpenVDBLink*)


LoadOpenVDBLink[] :=
    Module[{deps},
        deps = FileNameJoin[{$libraryDirectory, "dependencies.m"}];
        Check[
            If[FileExistsQ[deps], Get[deps]],
            Return[$Failed]
        ];

        If[Quiet @ LoadTemplate[$OpenVDBTemplate] === $Failed,
            Return[$Failed]
        ];
    ]


(* ::Subsection::Closed:: *)
(*Call to LoadOpenVDBLink*)


If[!TrueQ[$templateLoaded],
    If[LoadOpenVDBLink[] === $Failed,
        OpenVDBLink`Developer`Recompile[]
    ];
    $templateLoaded = True;
]


(* ::Section:: *)
(*Unit Testing*)


(* ::Subsection::Closed:: *)
(*TestOpenVDBLink*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBLink`Developer`TestOpenVDBLink[iareas_:All] :=
    Block[{wlts, areas, filepattern, torun, results, final, passcnt, failcnt, percentage, totaltime, stats},
        wlts = FileNames[$wltFilePattern];
        areas = gridAreas[iareas];
        (
            filepattern = Alternatives @@ areas;
            torun = Select[wlts, StringMatchQ[FileBaseName[#], filepattern]&];

            results = Map[
                (PrintTemporary[FileBaseName[#]];
                TestReport[#, SameTest -> sameExpression])&,
                torun
            ];
            (
                final = AssociationThread[FileBaseName /@ torun, results];

                passcnt = Total[#["TestsSucceededCount"]& /@ results];
                failcnt = Total[#["TestsFailedCount"]& /@ results];
                percentage = N[Quiet[Divide[passcnt, passcnt + failcnt]]];
                totaltime = Total[#["TimeElapsed"]& /@ results];

                stats = Association[{
                    "TestsSucceededCount" -> passcnt,
                    "TestsFailedCount" -> failcnt,
                    "SuccessRate" -> percentage,
                    "TimeElapsed" -> totaltime
                }];

                Association[{"TestResults" -> final, "GlobalStatistics" -> stats}]

            ) /; MatchQ[results, {__TestReportObject}]

        ) /; VectorQ[areas, StringQ]
    ]


OpenVDBLink`Developer`TestOpenVDBLink[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


$wltFilePattern = FileNameJoin[{$unitTestDirectory, "wlt", "*.wlt"}];


$testAreaList = FileBaseName /@ FileNames[$wltFilePattern];


gridAreas[types_] := DeleteDuplicates[Flatten[iGridAreas /@ Flatten[{types}]]]


iGridAreas[All] := $testAreaList
iGridAreas[area_] /; MemberQ[$testAreaList, area] := area
iGridAreas[___] = $Failed;


(* ::Text:: *)
(*SameQ with a fuzzy tolerance on inexact numbers. The idea is that machine numbers can vary across machines.*)
(*Assumes no evaluation leaks can happen and ignores Orderless pattern matching.*)


sameExpression[expr1_, expr2_] :=
    Block[{Internal`$EqualTolerance = 11, Internal`$SameQTolerance = 11},
        expr1 === expr2
    ]


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBLink`Developer`TestOpenVDBLink] = {"ArgumentsPattern" -> {_.}};


addCodeCompletion["OpenVDBLink`Developer`TestOpenVDBLink"][iGridAreas[All]];


(* ::Subsection::Closed:: *)
(*WLTToNotebook*)


(* ::Text:: *)
(*Specialized version of https://resources.wolframcloud.com/FunctionRepository/resources/WLTToNotebook/*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBLink`Developer`WLTToNotebook[unitTestName_] := Enclose @ Module[{
    file,
    heldContents,
    cellids,
    cells
},
    file = FileNameJoin[{
        $OpenVDBInstallationDirectory,
        "UnitTests", "wlt",
        FileBaseName[unitTestName] <> ".wlt"
    }];
    ConfirmBy[file, FileExistsQ, "File does not exist"];
    ConfirmBy[ToLowerCase @ FileExtension[file], MatchQ["wlt" | "mt"], "File extension is not .wlt or .mt"];

    Block[{$Context, $ContextPath, noTitleYetQ = True},
        Needs["MUnit`"];
        heldContents = Confirm[Import[file, {"WL", "HeldExpressions"}], "Import error"];
        cellids = CreateDataStructure["HashSet"];
        heldContents = testToCellGroup[#, cellids]& /@ heldContents;
    ];

    cells = Cases[Flatten @ heldContents, _Cell];

    cells = createCellGroup[cells, "Section"];
    cells = Replace[cells, CellGroupData[l:{_[_, "Section"], __}, c___] :> CellGroupData[createCellGroup[l, "Subsection"], c], {1}];

    NotebookPut @ Notebook[
        cells,
        ShowGroupOpener -> True,
        TaggingRules -> Association["$testsRun" -> False],
        StyleDefinitions -> FrontEnd`FileName[
            {"MUnit"}, "MUnit.nb",
            CharacterEncoding -> "UTF-8"
        ]
    ]
];


(* ::Subsubsection::Closed:: *)
(*Utilities*)


generateUniqueID[max_, hashTable_] := Module[{i = 0},
    TimeConstrained[
        While[True,
            i = RandomInteger[max];
            If[ TrueQ @ hashTable["Insert", i],
                Break[]
            ]
        ],
        2
    ];
    i
];


SetAttributes[testToCellGroup, HoldAllComplete];


(* Handle verification tests terminated by a ; *)
testToCellGroup[
    HoldComplete[CompoundExpression[expressions__]],
    rest___
] := Map[
    testToCellGroup[#, rest]&,
    Thread @ HoldComplete[{expressions}]
];

(* Handle 1-arg tests *)
testToCellGroup[
    HoldComplete[test : VerificationTest[fst_, {}, args___]],
    cellids_
] /; Quiet @ CheckArgs[test, 1] := testToCellGroup[VerificationTest[fst, {}, {}], cellids];

(* Handle 1-arg tests *)
testToCellGroup[
    HoldComplete[test : VerificationTest[fst_, args___]],
    cellids_
] /; Quiet @ CheckArgs[test, 1] := testToCellGroup[VerificationTest[fst, True, {}, args], cellids];

(* Handle 2-arg tests *)
testToCellGroup[
    HoldComplete[test : VerificationTest[fst_, snd_, args___]],
    cellids_
] /; Quiet @ CheckArgs[test, 2] := testToCellGroup[VerificationTest[fst, snd, {}, args], cellids];

(* Handle 3-arg tests *)
testToCellGroup[
    HoldComplete[test_VerificationTest],
    cellids_
] /; Quiet @ CheckArgs[test, 3] := testToCellGroup[test, cellids]

(* Convert test to Cells *)
testToCellGroup[
    test : VerificationTest[in_, out_, msgs_, opts___],
    cellids_
] := With[{
    imax = 10^9
},
    Cell @ CellGroupData[
        {
            Cell[
                BoxData @ MakeBoxes[in, StandardForm],
                "VerificationTest",
                CellID -> generateUniqueID[imax, cellids]
            ],
            Cell[
                BoxData @ MakeBoxes[out, StandardForm],
                "ExpectedOutput",
                CellID -> generateUniqueID[imax, cellids]
            ],
            Cell[
                BoxData @ MakeBoxes[msgs, StandardForm],
                "ExpectedMessage",
                CellID -> generateUniqueID[imax, cellids]
            ],
            Cell[
                BoxData @ MakeBoxes[{opts}, StandardForm],
                "TestOptions",
                CellID -> generateUniqueID[imax, cellids]
            ],
            Cell[
                BoxData @ ToBoxes @ MUnit`bottomCell[],
                "BottomCell",
                CellID -> generateUniqueID[imax, cellids]
            ]
        },
        Open
    ]
];

testToCellGroup[HoldComplete[MUnit`BeginTestSection[section_String]], _] := Cell[section, sectionType[section]];

testToCellGroup[other_, _] := Nothing;


sectionType[title_] /; noTitleYetQ := (noTitleYetQ = False; "Title");
sectionType[symbol_String] /; StringStartsQ[symbol, "Initialization" | (("$"...) ~~ "OpenVDB")] = "Subsection";
sectionType[_] = "Section";


createCellGroup[cells_, section_] :=
    Block[{splitcells},
        splitcells = SplitBy[cells, #[[-1]] === section&];
        (
            Prepend[
                CellGroupData[Join[##], Closed]& @@@ Partition[Rest[splitcells], 2],
                splitcells[[1, 1]]
            ]

        ) /; OddQ[Length[splitcells]] && MatchQ[splitcells, {{_}, _, _, ___}]
    ];


createCellGroup[args___] := $Failed


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBLink`Developer`WLTToNotebook] = {"ArgumentsPattern" -> {_}};


addCodeCompletion["OpenVDBLink`Developer`WLTToNotebook"][iGridAreas[All]];


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*addCodeCompletion*)


(* ::Text:: *)
(*https://resources.wolframcloud.com/FunctionRepository/resources/AddCodeCompletion*)


PackageScope["addCodeCompletion"]


addCodeCompletion[sym_Symbol][args___] := addCodeCompletion[SymbolName[sym]][args]


addCodeCompletion[function_String][args___] :=
    With[{processed = {args} /. {
            None -> 0, "AbsoluteFileName" -> 2, "RelativeFileName" -> 3,
            "Color" -> 4, "PackageName" -> 7, "DirectoryName" -> 8,
            "InterpreterType" -> 9}
        },
        FE`Evaluate[FEPrivate`AddSpecialArgCompletion[function -> processed]];
    ]


(* ::Subsection::Closed:: *)
(*registerForLevelSet*)


(* ::Subsubsection::Closed:: *)
(*Main*)


PackageScope["registerForLevelSet"]


registerForLevelSet[func_Symbol, pos_:{{}}] :=
    Block[{},
        func[args__] :=
            With[{res = tryToLevelSet[pos, func, args]},
                res /; res =!= $Failed
            ]
    ]


(* ::Subsubsection::Closed:: *)
(*level set dispatch*)


tryToLevelSet[pos_, func_, args__] /; validTryToLevelSetQ[] :=
    Block[{levelsets, reps, res, $inToLevelSet = True},
        levelsets = If[pos === {{}},
            tryOpenVDBLevelSet /@ {args},
            Quiet @ Extract[{args}, pos, tryOpenVDBLevelSet]
        ];
        (
            reps = Which[
                pos === {{}}, pos -> levelsets,
                ListQ[levelsets], Thread[pos -> levelsets],
                True, pos -> levelsets
            ];

            res = func @@ ReplacePart[{args}, reps];

            res /; res =!= $Failed

        ) /; validScalarGridCollectionQ[levelsets]
    ]

tryToLevelSet[___] := $Failed


(* ::Subsubsection::Closed:: *)
(*tryOpenVDBLevelSet*)


tryOpenVDBLevelSet[vdb_?OpenVDBScalarGridQ] := vdb
tryOpenVDBLevelSet[_?OpenVDBGridQ] = $Failed;
tryOpenVDBLevelSet[{file_String?FileExistsQ, name_}] :=
    Block[{vdb},
        vdb = OpenVDBImport[file, name];
        If[OpenVDBScalarGridQ[vdb],
            vdb,
            $Failed
        ]
    ]
tryOpenVDBLevelSet[file_String] := tryOpenVDBLevelSet[{file, Automatic}]
tryOpenVDBLevelSet[{File[file_], name_}] := tryOpenVDBLevelSet[{file, name}]
tryOpenVDBLevelSet[File[file_]] := tryOpenVDBLevelSet[{file, Automatic}]
tryOpenVDBLevelSet[opt_?OptionQ] := opt
tryOpenVDBLevelSet[expr_] :=
    With[{res = Quiet @ OpenVDBLevelSet[expr]},
        res /; OpenVDBScalarGridQ[res]
    ]
tryOpenVDBLevelSet[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


$inToLevelSet = False;


validTryToLevelSetQ[] := TrueQ[!$inToLevelSet] && TrueQ[Positive[$OpenVDBSpacing]] && TrueQ[Positive[$OpenVDBHalfWidth]]


validScalarGridCollectionQ[levelsets_List] := Length[levelsets] > 0 && VectorQ[levelsets, validScalarGridCollectionQ]
validScalarGridCollectionQ[expr_] := OpenVDBScalarGridQ[expr] || OptionQ[expr]


(* ::Subsection::Closed:: *)
(*Argument checking*)


PackageScope["CheckArgs"]


SetAttributes[{CheckArgs, catchMessages}, HoldFirst]


CheckArgs[expr_, spec_] := !FailureQ[catchMessages[System`Private`Arguments[expr, spec, HoldComplete, {}]]]


catchMessages[expr_] := Catch[Internal`HandlerBlock[{"Message", Throw[$Failed, "iArgumentsMessage"]&}, expr], "iArgumentsMessage"];


(* ::Subsection::Closed:: *)
(*halfWidth*)


PackageScope["halfWidth"]


halfWidth[vdb_] := vdb["getHalfwidth"[]]


(* ::Subsection::Closed:: *)
(*emptyVDBQ*)


PackageScope["emptyVDBQ"]


emptyVDBQ[vdb_] := TrueQ[vdb["getIsEmpty"[]]]


(* ::Subsection::Closed:: *)
(*voxelSize*)


PackageScope["voxelSize"]


voxelSize[vdb_] := vdb["getVoxelSize"[]]


(* ::Subsection::Closed:: *)
(*Grid class*)


PackageScope["levelSetQ"]
PackageScope["fogVolumeQ"]
PackageScope["unknownClassQ"]


PackageScope["gridClassID"]
PackageScope["gridClassName"]


PackageScope["$gridUnknown"]
PackageScope["$gridLevelSet"]
PackageScope["$gridFogVolume"]


$gridUnknown = None;
$gridLevelSet = "LevelSet";
$gridFogVolume = "FogVolume";


$gridClassTypes = {$gridUnknown, $gridLevelSet, $gridFogVolume};
$gridClassIDs = Range[0, Length[$gridClassTypes]-1];


MapThread[(gridClassID[#1] = #2)&, {$gridClassTypes, $gridClassIDs}];
gridClassID[_] = gridClassID[$gridUnknown];


MapThread[(gridClassName[#1] = #2)&, {$gridClassIDs, $gridClassTypes}];
gridClassName[_] = gridClassName[0];


(* ::Subsection::Closed:: *)
(*Resampling methods*)


PackageScope["resamplingMethod"]


resamplingMethod["Nearest" | 0] = 0;
resamplingMethod[Automatic | "Linear" | 1] = 1;
resamplingMethod["Quadratic" | 2] = 2;
resamplingMethod[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Space regime*)


PackageScope["$indexregime"]
PackageScope["$worldregime"]
PackageScope["regimeQ"]
PackageScope["indexRegimeQ"]
PackageScope["worldRegimeQ"]


$indexregime = "Index";
$worldregime = "World";


regimeQ[$indexregime] = True;
regimeQ[$worldregime] = True;
regimeQ[_] = False;


indexRegimeQ[$indexregime] = True;
indexRegimeQ[_] = False;


worldRegimeQ[$worldregime] = True;
worldRegimeQ[_] = False;


PackageScope["regimeConvert"]


regimeConvert[vdb_, vals_, fromregime_ -> toregime_, dim_:1] :=
    Which[
        indexRegimeQ[fromregime] && fromregime === toregime,
            Round[vals],
        fromregime === toregime,
            vals,
        indexRegimeQ[fromregime], (* convert index space values to world space values *)
            vals * voxelSize[vdb]^dim,
        True,                     (* convert world space values to index space values *)
            Round[Divide[vals, voxelSize[vdb]^dim]]
    ]


(* ::Subsection::Closed:: *)
(*realQ*)


PackageScope["realQ"]


realQ = Internal`RealValuedNumericQ;


(* ::Subsection::Closed:: *)
(*Bounds & coordinates*)


PackageScope["intervalQ"]


intervalQ[int_] := VectorQ[int, realQ] && Length[int] === 2 && LessEqual @@ int


PackageScope["bounds2DQ"]
PackageScope["bounds3DQ"]


bounds2DQ[bds_] := MatrixQ[bds, realQ] && Dimensions[bds] === {2, 2} && And @@ LessEqual @@@ bds


bounds3DQ[bds_] := MatrixQ[bds, realQ] && Dimensions[bds] === {3, 2} && And @@ LessEqual @@@ bds


PackageScope["coordinatesQ"]
PackageScope["coordinateQ"]


coordinatesQ[coords_] := MatrixQ[coords, realQ] && Length[coords[[1]]] == 3
coordinatesQ[___] = False;


coordinateQ[coords_] := VectorQ[coords, realQ] && Length[coords] == 3
coordinateQ[___] = False;


(* ::Subsection::Closed:: *)
(*gridIsoValue*)


PackageScope["gridIsoValue"]


gridIsoValue[x_?realQ, _] := x
gridIsoValue[Automatic, _?fogVolumeQ] = 0.5;
gridIsoValue[Automatic, _] = 0.0;
gridIsoValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Same structures*)


PackageScope["sameGridTypeQ"]
PackageScope["sameVoxelSizeQ"]


sameGridTypeQ[vdbs__?OpenVDBGridQ] := SameQ @@ (#["GridType"]& /@ {vdbs})


sameGridTypeQ[___] = False;


sameVoxelSizeQ[vdbs__?OpenVDBGridQ] := SameQ @@ (#["VoxelSize"]& /@ {vdbs})


sameVoxelSizeQ[___] = False;


(* ::Subsection::Closed:: *)
(*internal GridQ functions*)


PackageScope["levelSetQ"]
PackageScope["fogVolumeQ"]
PackageScope["unknownClassQ"]


PackageScope["numericGridQ"]
PackageScope["carefulNumericGridQ"]
PackageScope["nonMaskGridQ"]
PackageScope["carefulNonMaskGridQ"]


PackageScope["pixelGridQ"]
PackageScope["carefulPixelGridQ"]


With[{lid = gridClassID[$gridLevelSet], fid = gridClassID[$gridFogVolume]},
    levelSetQ[vdb_] := vdb["getGridClass"[]] === lid;
    fogVolumeQ[vdb_] := vdb["getGridClass"[]] === fid;
    unknownClassQ[vdb_] := vdb["getGridClass"[]] =!= lid && vdb["getGridClass"[]] =!= fid;
]


nonMaskGridQ[vdb_] := vdb[[2]] =!= $maskType


carefulNonMaskGridQ[vdb_] := OpenVDBGridQ[vdb] && nonMaskGridQ[vdb]


pixelGridQ[vdb_] := pixelTypeQ[vdb[[2]]]


carefulPixelGridQ[vdb_] := OpenVDBGridQ[vdb] && pixelGridQ[vdb]


(* ::Section:: *)
(*Documentation*)


(* ::Subsection::Closed:: *)
(*OpenVDBDocumentation*)


(* ::Subsubsection::Closed:: *)
(*OpenVDBDocumentation*)


OpenVDBDocumentation[args___] /; !CheckArgs[OpenVDBDocumentation[args], {0, 1}] = $Failed;


OpenVDBDocumentation[args___] :=
    With[{res = iOpenVDBDocumentation[args]},
        res /; res =!= $Failed
    ]


OpenVDBDocumentation[args___] := mOpenVDBDocumentation[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDocumentation*)


iOpenVDBDocumentation[] := iOpenVDBDocumentation["Notebook"]


iOpenVDBDocumentation["Web"] := SystemOpen[$vdbWebDocURL]


iOpenVDBDocumentation["Notebook"] := notebookDocumentation[]


iOpenVDBDocumentation["Update"] :=
    (
        Quiet[DeleteFile[$vdbDoc]];
        notebookDocumentation[]
    )


iOpenVDBDocumentation[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBDocumentation] = {"ArgumentsPattern" -> {_.}};


addCodeCompletion[OpenVDBDocumentation][{"Web", "Notebook", "Update"}];


(* ::Subsubsection::Closed:: *)
(*notebookDocumentation*)


notebookDocumentation[] /; FileExistsQ[$vdbDoc] := NotebookOpen[$vdbDoc]


notebookDocumentation[] :=
    Block[{zip},
        If[!FileExistsQ[$vdbDocDir],
            CreateDirectory[$vdbDocDir];
        ];

        zip = URLDownload[$vdbDocURL, $vdbDocDir];
        (
            Quiet[
                ExtractArchive[zip, $vdbDocDir, FileNameTake[$vdbDoc]];
                DeleteFile[zip];
            ];

            NotebookOpen[$vdbDoc] /; FileExistsQ[$vdbDoc]

        ) /; FileExistsQ[zip]
    ]


notebookDocumentation[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Documentation paths and URLs*)


$vdbDocDir = FileNameJoin[{$UserBaseDirectory, "ApplicationData", "OpenVDBLink"}];


$vdbDoc = FileNameJoin[{$vdbDocDir, "OpenVDBLink.nb"}];


$vdbDocURL = "https://www.openvdb.org/download/files/OpenVDBLink.nb.zip";


$vdbWebDocURL = "https://www.openvdb.org/documentation/wolfram";


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDocumentation[type_] :=
    (
        If[type =!= "Web" && type =!= "Notebook",
            Message[OpenVDBDocumentation::type, type, 1];
        ];

        $Failed
    )


mOpenVDBDocumentation[___] = $Failed;


OpenVDBDocumentation::type = "`1` at position `2` is not one of \"Web\", \"Notebook\", or \"Update\".";
