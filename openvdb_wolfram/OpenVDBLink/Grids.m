(* ::Package:: *)

(* ::Title:: *)
(*Grids*)


(* ::Subtitle:: *)
(*Basic operations on OpenVDB Grids.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageImport["OpenVDBLink`LTemplate`"]


PackageExport["OpenVDBGrid"]
PackageExport["OpenVDBGridQ"]
PackageExport["OpenVDBGrids"]
PackageExport["OpenVDBGridTypes"]
PackageExport["OpenVDBScalarGridQ"]
PackageExport["OpenVDBIntegerGridQ"]
PackageExport["OpenVDBVectorGridQ"]
PackageExport["OpenVDBBooleanGridQ"]
PackageExport["OpenVDBMaskGridQ"]
PackageExport["OpenVDBCreateGrid"]
PackageExport["OpenVDBClearGrid"]
PackageExport["OpenVDBCopyGrid"]
PackageExport["$OpenVDBSpacing"]
PackageExport["$OpenVDBHalfWidth"]
PackageExport["$OpenVDBCreator"]
PackageExport["OpenVDBDefaultSpace"]


OpenVDBGrid::usage = "OpenVDBGrid[id, type] represents an instance of an OpenVDB grid of a given type.";
OpenVDBGridQ::usage = "OpenVDBGridQ[expr] returns True if expr represents an active instance of an OpenVDB grid.";
OpenVDBGrids::usage = "OpenVDBGrids[] returns a list of active OpenVDB grids.";
OpenVDBGridTypes::usage = "OpenVDBGridTypes[] returns a list of all valid grid types.";


OpenVDBScalarGridQ::usage = "OpenVDBScalarGridQ[expr] returns True if expr represents an active instance of an OpenVDB scalar grid.";
OpenVDBIntegerGridQ::usage = "OpenVDBIntegerGridQ[expr] returns True if expr represents an active instance of an OpenVDB integer grid.";
OpenVDBVectorGridQ::usage = "OpenVDBVectorGridQ[expr] returns True if expr represents an active instance of an OpenVDB vector grid.";
OpenVDBBooleanGridQ::usage = "OpenVDBBooleanGridQ[expr] returns True if expr represents an active instance of an OpenVDB Boolean grid.";
OpenVDBMaskGridQ::usage = "OpenVDBMaskGridQ[expr] returns True if expr represents an active instance of an OpenVDB mask grid.";


OpenVDBCreateGrid::usage = "OpenVDBCreateGrid[] creates an instance of an OpenVDB grid.";
OpenVDBClearGrid::usage = "OpenVDBClearGrid[expr] clears all data in an OpenVDB grid, effectively leaving the grid empty.";
OpenVDBCopyGrid::usage = "OpenVDBCopyGrid[expr] creates a copy of an OpenVDB grid.";


$OpenVDBSpacing::usage = "$OpenVDBSpacing is a global variable that represents the default voxel spacing.";
$OpenVDBHalfWidth::usage = "$OpenVDBHalfWidth is a global variable that represents the default signed distance field half width. The default value is 3.0.";
$OpenVDBCreator::usage = "$OpenVDBCreator is a global variable that fills the creator metadata when a grid is created. The default value is None.";


OpenVDBDefaultSpace::usage = "OpenVDBDefaultSpace[f] returns the default coordinate space (\"Index\" or \"World\") of the OpenVDBLink function f, or Missing[\"NotApplicable\"] if f does not need a coordinate space.";


(* ::Section:: *)
(*OpenVDB grids*)


(* ::Subsection::Closed:: *)
(*OpenVDBGrid*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGrid[i_, type_][(func_String)[args___]] := typeClass[type][i][func[args]]


(* ::Subsubsection::Closed:: *)
(*Formatting*)


OpenVDBGrid /: MakeBoxes[vdb_OpenVDBGrid?OpenVDBGridQ, fmt_] :=
    Block[{lvlsetQ, class, infos, icon},
        lvlsetQ = levelSetQ[vdb];

        infos = {
            If[StringQ[vdb["Name"]], {BoxForm`SummaryItem[{"Name: ", vdb["Name"]}]}, Nothing],
            {BoxForm`SummaryItem[{"Class: ", formatGridClass[vdb]}]},
            {BoxForm`SummaryItem[{"Type: ", formatTreeType[vdb["GridType"]]}]},
            {BoxForm`SummaryItem[{"Voxel size: ", voxelSize[vdb]}]},
            If[lvlsetQ, {BoxForm`SummaryItem[{"Half width: ", vdb["HalfWidth"]}]}, Nothing],
            {BoxForm`SummaryItem[{"Expression ID: ", vdb[[1]]}]}
        };

        icon = Which[
            lvlsetQ,
                $lvlseticon,
            fogVolumeQ[vdb],
                $fogvolicon,
            OpenVDBScalarGridQ[vdb],
                $scalargridicon,
            OpenVDBVectorGridQ[vdb],
                $vecgridicon,
            OpenVDBIntegerGridQ[vdb],
                $intgridicon,
            OpenVDBBooleanGridQ[vdb],
                $boolgridicon,
            OpenVDBMaskGridQ[vdb],
                $maskgridicon,
            True,
                None
        ];

        BoxForm`ArrangeSummaryBox[
            OpenVDBGrid,
            vdb,
            icon,
            infos[[1 ;; 2]],
            infos[[3 ;; -1]],
            fmt,
            "Interpretable" -> Automatic
        ]
    ]


formatGridClass[vdb_?levelSetQ] = "Level set";
formatGridClass[vdb_?fogVolumeQ] = "Fog volume";
formatGridClass[vdb_?OpenVDBVectorGridQ] = $vectorType;
formatGridClass[vdb_?OpenVDBIntegerGridQ] = $integerType;
formatGridClass[vdb_?OpenVDBScalarGridQ] = $scalarType;
formatGridClass[vdb_?OpenVDBBooleanGridQ] = $booleanType;
formatGridClass[vdb_?OpenVDBMaskGridQ] = $maskType;
formatGridClass[_] = None;


formatTreeType[type_String] /; StringMatchQ[type, "Tree_" ~~ t__ ~~ ("_" ~~ (DigitCharacter..)).. /; StringFreeQ[t, "_"]] :=
    Block[{split, t, dig},
        split = StringSplit[type, "_"];
        t = split[[2]];
        dig = split[[3 ;; -1]];

        StringRiffle[dig, {t <> " (", ",", ")"}]
    ]
formatTreeType[expr_] := expr


(* ::Subsection::Closed:: *)
(*OpenVDBGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGridQ[vdb:OpenVDBGrid[_, type_]] := ManagedLibraryExpressionQ[vdb, typeGridName[type]]


OpenVDBGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*OpenVDBGrids*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGrids[args___] /; !CheckArgs[OpenVDBGrids[args], {0, 1}] = $Failed;


OpenVDBGrids[args___] :=
    With[{res = iOpenVDBGrids[args]},
        res /; res =!= $Failed
    ]


OpenVDBGrids[args___] := mOpenVDBGrids[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBGrids*)


iOpenVDBGrids[] := iOpenVDBGrids[Automatic]


iOpenVDBGrids[Automatic] := Select[iOpenVDBGrids[All], Length[#] > 0&]


iOpenVDBGrids[All] :=
    With[{typesnoaliases = Join @@ Keys /@ $GridClassData},
        Association[# -> iOpenVDBGrids[#]& /@ typesnoaliases]
    ]


iOpenVDBGrids[type_String?aliasTypeQ] := iOpenVDBGrids[resolveAliasType[type]]


iOpenVDBGrids[type_String] :=
    Block[{gridlist},
        gridlist = Quiet @ LExpressionList[typeGridName[type]];

        If[ListQ[gridlist],
            OpenVDBGrid[#, type]& @@@ gridlist,
            $Failed
        ]
    ]


iOpenVDBGrids[types:{___String}] :=
    With[{grids = iOpenVDBGrids /@ types},
        AssociationThread[types, grids] /; FreeQ[grids, $Failed]
    ]


iOpenVDBGrids[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBGrids] = {"ArgumentsPattern" -> {_.}};


addCodeCompletion[OpenVDBGrids][$gridTypeList];


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBGrids[Automatic|All] = $Failed;


mOpenVDBGrids[expr_List] /; Complement[expr, OpenVDBGridTypes[]] =!= {} :=
    (
        Message[OpenVDBGrids::type2, expr];
        $Failed
    )


mOpenVDBGrids[expr_] /; !MemberQ[OpenVDBGridTypes[], expr] :=
    (
        Message[OpenVDBGrids::type, expr];
        $Failed
    )


mOpenVDBGrids[___] = $Failed;


OpenVDBGrids::type = "`1` is not a supported grid type. Evaluate OpenVDBGridTypes[] to see the list of supported types.";
OpenVDBGrids::type2 = "`1` is not a list of supported grid types. Evaluate OpenVDBGridTypes[] to see the list of supported types.";


(* ::Subsection::Closed:: *)
(*OpenVDBGridTypes*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGridTypes[args___] /; !CheckArgs[OpenVDBGridTypes[args], {0, 1}] = $Failed;


OpenVDBGridTypes[args___] :=
    With[{res = iOpenVDBGridTypes[args]},
        res /; res =!= $Failed
    ]


OpenVDBGridTypes[args___] := mOpenVDBGridTypes[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBGridTypes*)


Scan[(iOpenVDBGridTypes[#] = Join[DeleteMissing[Values[$GridClassData[#]][[All, "Alias"]]], Keys[$GridClassData[#]]])&, $classTypeList]


iOpenVDBGridTypes[All] = $gridTypeList;


iOpenVDBGridTypes[] = iOpenVDBGridTypes[All];


iOpenVDBGridTypes[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBGridTypes] = {"ArgumentsPattern" -> {_.}};


addCodeCompletion[OpenVDBGridTypes][$classTypeList];


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBGridTypes[All] = $Failed;


mOpenVDBGridTypes[expr_] /; !MemberQ[$classTypeList, expr] :=
    (
        Message[OpenVDBGridTypes::type, expr];
        $Failed
    )


mOpenVDBGridTypes[___] = $Failed;


OpenVDBGridTypes::type = "`1` is not a supported family of grids.";


(* ::Section:: *)
(*Specific grids*)


(* ::Subsection::Closed:: *)
(*OpenVDBScalarGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBScalarGridQ[vdb:OpenVDBGrid[_, type_]] := OpenVDBGridQ[vdb] && !MissingQ[$GridClassData[$scalarType, type]]


OpenVDBScalarGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBScalarGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*OpenVDBIntegerGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBIntegerGridQ[vdb:OpenVDBGrid[_, type_]] := OpenVDBGridQ[vdb] && !MissingQ[$GridClassData[$integerType, type]]


OpenVDBIntegerGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBIntegerGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*OpenVDBVectorGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBVectorGridQ[vdb:OpenVDBGrid[_, type_]] := OpenVDBGridQ[vdb] && !MissingQ[$GridClassData[$vectorType, type]]


OpenVDBVectorGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBVectorGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*OpenVDBBooleanGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBBooleanGridQ[vdb:OpenVDBGrid[_, $booleanType]] := OpenVDBGridQ[vdb]


OpenVDBBooleanGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBBooleanGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*OpenVDBMaskGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBMaskGridQ[vdb:OpenVDBGrid[_, $maskType]] := OpenVDBGridQ[vdb]


OpenVDBMaskGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBMaskGridQ] = {"ArgumentsPattern" -> {_}};


(* ::Section:: *)
(*Constructor & destructor*)


(* ::Subsection::Closed:: *)
(*OpenVDBCreateGrid*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBCreateGrid] = {"BackgroundValue" -> Automatic, "Creator" :> $OpenVDBCreator, "GridClass" -> None, "Name" -> None};


OpenVDBCreateGrid[args___] /; !CheckArgs[OpenVDBCreateGrid[args], {0, 2}] = $Failed;


OpenVDBCreateGrid[args___] :=
    With[{res = iOpenVDBCreateGrid[args]},
        res /; res =!= $Failed
    ]


OpenVDBCreateGrid[args___] := mOpenVDBCreateGrid[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBCreateGrid*)


Options[iOpenVDBCreateGrid] = Options[OpenVDBCreateGrid];


iOpenVDBCreateGrid[opts:OptionsPattern[]] := iOpenVDBCreateGrid[$OpenVDBSpacing, $gridTypeList[[1]], opts]


iOpenVDBCreateGrid[spacing_?Positive, opts:OptionsPattern[]] := iOpenVDBCreateGrid[spacing, $gridTypeList[[1]], opts]


iOpenVDBCreateGrid[spacing_?Positive, type_String, opts:OptionsPattern[]] :=
    Block[{vdb = newVDB[type]},
        (
            setVDBProperties[vdb, "VoxelSize" -> spacing, opts]

        ) /; OpenVDBGridQ[vdb]
    ]


iOpenVDBCreateGrid[ovdb_?OpenVDBGridQ, opts:OptionsPattern[]] :=
    Block[{vdb = newVDB[ovdb[[2]]], vdbprops},
        vdbprops = OpenVDBProperty[ovdb, {"VoxelSize", "BackgroundValue", "Creator", "GridClass", "Name"}, "RuleList"];
        (
            setVDBProperties[vdb, opts, Sequence @@ vdbprops]

        ) /; OpenVDBGridQ[vdb]
    ]


iOpenVDBCreateGrid[___] = $Failed;


newVDB[type_?aliasTypeQ] := newVDB[resolveAliasType[type]]


newVDB[type_String] :=
    With[{res = Quiet @ CreateManagedLibraryExpression[typeGridName[type], OpenVDBGrid[#, type]&]},
        res /; res =!= $Failed
    ]


newVDB[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*setVDBProperties*)


Options[setVDBProperties] = Join[Options[OpenVDBCreateGrid], {"VoxelSize" :> $OpenVDBSpacing}];


setVDBProperties[vdb_, OptionsPattern[]] :=
    Block[{bg, creator, gclass, name, spacing},
        bg = OptionValue["BackgroundValue"];
        creator = OptionValue["Creator"];
        gclass = OptionValue["GridClass"];
        name = OptionValue["Name"];
        spacing = OptionValue["VoxelSize"];

        OpenVDBSetProperty[
            vdb,
            {
                If[bg === Automatic, Nothing, "BackgroundValue" -> bg],
                "Creator" -> creator,
                "GridClass" -> gclass,
                "Name" -> name,
                "VoxelSize" -> spacing
            }
        ];

        vdb
    ]


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBCreateGrid] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}};


addCodeCompletion[OpenVDBCreateGrid][None, $gridTypeList];


(* ::Subsubsection::Closed:: *)
(*Messages*)


Options[mOpenVDBCreateGrid] = Options[OpenVDBCreateGrid];


mOpenVDBCreateGrid[OptionsPattern[]] /; !TrueQ[$OpenVDBSpacing > 0] :=
    (
        Message[OpenVDBCreateGrid::novoxsz];
        $Failed
    );


mOpenVDBCreateGrid[vx_, ___] /; !TrueQ[vx > 0] && !OptionQ[vx] :=
    (
        Message[OpenVDBCreateGrid::nonpos, vx, 1];
        $Failed
    )


mOpenVDBCreateGrid[_, type_, ___] /; !MemberQ[$gridTypeList, type] && !OptionQ[type] :=
    (
        Message[OpenVDBCreateGrid::type, type];
        $Failed
    )


mOpenVDBCreateGrid[___] = $Failed;


OpenVDBCreateGrid::novoxsz = "No grid spacing is provided since $OpenVDBSpacing is not a positive number.";
OpenVDBCreateGrid::nonpos = "`1` at position `2` is expected to be a positive number";


OpenVDBCreateGrid::type = "`1` is not a supported grid type. Evaluate OpenVDBGridTypes[] to see the list of supported types."


(* ::Subsection::Closed:: *)
(*OpenVDBClearGrid*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBClearGrid[args___] /; !CheckArgs[OpenVDBClearGrid[args], 1] = $Failed;


OpenVDBClearGrid[args___] :=
    With[{res = iOpenVDBClearGrid[args]},
        res /; res =!= $Failed
    ]


OpenVDBClearGrid[args___] := mOpenVDBClearGrid[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBClearGrid*)


SetAttributes[iOpenVDBClearGrid, Listable];


iOpenVDBClearGrid[vdb_?OpenVDBGridQ] :=
    (
        vdb["createEmptyGrid"[]];
        vdb
    )


iOpenVDBClearGrid[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBClearGrid] = {"ArgumentsPattern" -> {_}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBClearGrid[expr_] /; !OpenVDBGridQ[expr] && !VectorQ[expr, OpenVDBGridQ] :=
    (
        Message[OpenVDBClearGrid::grids, expr];
        $Failed
    )


mOpenVDBClearGrid[___] = $Failed;


OpenVDBClearGrid::grids = "`1` is not a grid or list of grids.";


(* ::Subsection::Closed:: *)
(*OpenVDBCopyGrid*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBCopyGrid] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBCopyGrid[args___] /; !CheckArgs[OpenVDBCopyGrid[args], 1] = $Failed;


OpenVDBCopyGrid[args___] :=
    With[{res = iOpenVDBCopyGrid[args]},
        res /; res =!= $Failed
    ]


OpenVDBCopyGrid[args___] := mOpenVDBCopyGrid[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBCopyGrid*)


Options[iOpenVDBCopyGrid] = Options[OpenVDBCopyGrid];


iOpenVDBCopyGrid[vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    Block[{vdbcopy},
        vdbcopy = newVDB[vdb[[2]]];
        (
            vdbcopy["copyGrid"[vdb[[1]]]];

            OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdbcopy

        ) /; OpenVDBGridQ[vdbcopy]
    ]


iOpenVDBCopyGrid[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBCopyGrid] = {"ArgumentsPattern" -> {_, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBCopyGrid[expr_, ___] /; messageGridQ[expr, OpenVDBCopyGrid, False] = $Failed;


mOpenVDBCopyGrid[___] = $Failed;


(* ::Section:: *)
(*Default spacing, half width, and creator*)


(* ::Subsection::Closed:: *)
(*$OpenVDBSpacing*)


(* ::Text:: *)
(*OpenVDB uses default spacing of 1.0. We leave $OpenVDBSpacing initially unset to prevent confusion.*)


$OpenVDBSpacing /: SetDelayed[$OpenVDBSpacing, expr_] :=
    (
        OwnValues[$OpenVDBSpacing] = {HoldPattern[$OpenVDBSpacing] :> With[{w = expr}, w /; Positive[w]]};
    )


$OpenVDBSpacing /: Set[$OpenVDBSpacing, w_?Positive] :=
    (
        OwnValues[$OpenVDBSpacing] = {HoldPattern[$OpenVDBSpacing] :> w};
        w
    )

$OpenVDBSpacing /: Set[$OpenVDBSpacing, _] :=
    (
        Message[$OpenVDBSpacing::setpos];
        $Failed
    )


$OpenVDBSpacing::setpos = "$OpenVDBSpacing must be set to a positive number.";


(* ::Subsection::Closed:: *)
(*$OpenVDBHalfWidth*)


$OpenVDBHalfWidth /: SetDelayed[$OpenVDBHalfWidth, expr_] :=
    (
        OwnValues[$OpenVDBHalfWidth] = {HoldPattern[$OpenVDBHalfWidth] :> With[{w = expr}, w /; w >= 1.05]};
    )


$OpenVDBHalfWidth /: Set[$OpenVDBHalfWidth, w_ /; w > 0] :=
    (
        OwnValues[$OpenVDBHalfWidth] = {HoldPattern[$OpenVDBHalfWidth] :> w};
        w
    )

$OpenVDBHalfWidth /: Set[$OpenVDBHalfWidth, e_] :=
    (
        Message[$OpenVDBHalfWidth::setpos];
        $Failed
    );


$OpenVDBHalfWidth::setpos = "$OpenVDBHalfWidth must be set to a positive number.";


(* ::Text:: *)
(*OpenVDB typically uses 3.0 by default.*)


If[!ValueQ[$OpenVDBHalfWidth],
    $OpenVDBHalfWidth = 3.0
];


(* ::Subsection::Closed:: *)
(*$OpenVDBCreator*)


$OpenVDBCreator /: SetDelayed[$OpenVDBCreator, expr_] :=
    (
        OwnValues[$OpenVDBCreator] = {HoldPattern[$OpenVDBCreator] :> With[{c = expr}, c /; StringQ[c] || c === None]};
    )


$OpenVDBCreator /: Set[$OpenVDBCreator, c_ /; StringQ[c] || c === None] :=
    (
        OwnValues[$OpenVDBCreator] = {HoldPattern[$OpenVDBCreator] :> c};
        c
    )

$OpenVDBCreator /: Set[$OpenVDBCreator, _] :=
    (
        Message[$OpenVDBCreator::badset];
        $Failed
    );


$OpenVDBCreator::badset = "$OpenVDBCreator must be set to a string or None.";


If[!ValueQ[$OpenVDBCreator],
    $OpenVDBCreator = None
];


(* ::Subsection::Closed:: *)
(*OpenVDBDefaultSpace*)


OpenVDBDefaultSpace[_] = Missing["NotApplicable"];


OpenVDBDefaultSpace[___] = $Failed;


SyntaxInformation[OpenVDBDefaultSpace] = {"ArgumentsPattern" -> {_}};


(* ::Section:: *)
(*Icons*)


(* ::Subsection::Closed:: *)
(*Level Set*)


$lvlseticon = With[{
        pts1 = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}},
        pts2 = {{-1, 0}, {-0.5, -0.5}, {0, -1}, {0.5, -0.5}, {1, 0}, {0.5, 0.5}, {0, 1}, {-0.5, 0.5}},
        tfs = TranslationTransform /@ {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
    },
    Graphics[
        {
            AbsoluteThickness[0.5],
            {Darker[Green], BSplineCurve[#[.5pts1], SplineDegree -> 3, SplineClosed -> True]& /@ tfs},
            {Darker[Blend[{Red, Green}, 5./6]], BSplineCurve[#[.65pts1], SplineDegree -> 2, SplineClosed -> True]& /@ tfs},
            {Blend[{GrayLevel[0.2], Darker[Blend[{Red, Green}, 4./6]]}, 0.75], BSplineCurve[#[.82pts2], SplineDegree -> 3, SplineClosed -> True]& /@ tfs},
            {GrayLevel[0.2], BSplineCurve[pts1, SplineDegree -> 1, SplineClosed -> True]},
            {Blend[{GrayLevel[0.2], Darker[Blend[{Red, Green}, 2./6]]}, 0.75], BSplineCurve[.82pts2, SplineDegree -> 3, SplineClosed -> True]},
            {Darker[Blend[{Red, Green}, 1./6]], BSplineCurve[.65pts1, SplineDegree -> 2, SplineClosed -> True]},
            {Darker[Red], BSplineCurve[.5pts1, SplineDegree -> 3, SplineClosed -> True]}
        },
        ImageSize -> Dynamic[{Automatic, 3.5Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}],
        PlotRange -> {{-1, 1}, {-1, 1}}
    ]
];


(* ::Subsection::Closed:: *)
(*Fog Volume*)


$fogvolicon = Graphics[Raster[CompressedData["
1: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
"], {{0, 0}, {80., 80.}}, {0, 255}, ColorFunction -> GrayLevel],
ImageSize -> Dynamic[{Automatic, 3.5 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}],
PlotRange -> {{0, 80.}, {0, 80.}}
];


(* ::Subsection::Closed:: *)
(*Scalar*)


(* ::Input:: *)
(*ArrayPlot[Table[Sin[j^2+i],{i,Pi,0,-Pi/7},{j,0,Pi,Pi/7}],Mesh->All,MeshStyle->Directive[AbsoluteThickness[0.1],GrayLevel[0.2]],ColorFunction->"WL12DefaultVectorGradient",Frame->False]*)


$scalargridicon = Show[
    Uncompress["1:eJzdlXlQU1cUh4OggwgUKFRxq1ZlnHEphCUZtB7sYCU0RNlqoyMQ9EVjYgQeEURAplbKWNwtVMU0ghAWDQhuFAUmEoSiDCGshdIQINAEEUYWl1GaAM2rhDdmpv/1zrw795573nfPO/d33l0eesiPaUIgEFBTTbc1ghG2n7UHZc7SWrRmKguNZBprZ3M0nR8DjUQiEO3cdOrRjukd3/y0bZEKhCHum0RFMqAmhM+PdmiBl+F5TPf8Z3A0ln1vNbUF0oXVlvzWGiBMtEGQVTh+FDLaNTX/pw3CleI7J8mpvTr7MQm/rfl0H8yvIxb4KaVwa2GMvKamDdjD/K+jUsoBOV/KCws6ClvNbubYuip1fOFjV8GOIz06zrlZ4/7UA52QxAgZsGdJ4NezQRaU4i6df1kWs2ZjV4dePD8QGCRLZ4WevTV94yMBtVfPbvPwkgu9sFtnb3IeSyy3lEPC/PQn2cFiiG5stL63oBuWl4lPmH3XDDaPFCvPxBTBIlJtgPvSHl08JYkijnkTxvdaWB1MjWiAp+IlY9doN8C/cxUj/gDm/7z/Tv3n7d168ZAl6v1vw5V69uBab5Vbp378QnGp2GQnlofUN+Ux8hwp8La3vnA/mAF9ZY7xu4U9EPnUYnVmvwIE+/pFHoNV0GXVTDvvheVzsXpN09a9GH9UlO2VTr0Fh5jXjFvuhgL9bBWnxB87rxHaAXNCln48g7uWJ5kQ+kBNf1lmpjTT+VMKj8WTsrDzZcT9aNpS+RdwfOqpuTQZLDsR8cmS4ZZpPAIhC03khjKV8OSCUVByrxpuuvIrLw41QhYlZstiVKbjbxjnrBWvx/J57mkGtz0pHwierfyHsSHQTv6zoiHww/Ff3l6k/rZEP//JmdaXiBK5zl4Q9noTxaYTPIRc6xFBBVjPcbcel3aBwCHDsyO8FCQvL37/u3cU+DwW3U3egO27xqsMWfGHXI+vro16Q8pv19l7n/E3ujg2wM3j3cvYaB6EiVWZsWEf1o+kMnldfjmmf7+QePukKjXUnWLaPqhuhNPG7ZlJ1TKIu5oWLy6VwS8NryrsxnNg0Kd3V1pMDwQzZ1XZDDVBt116eDavEOqe7HSatw7bN8wijmZ1GztHW9EO4pmKPtifkbBJ/lk9WDUkBrT6toEgVV7SQemEZYyVUYs+lcC5eQP3uWL8+rULZDNeoANwTzXQ9CW9FcpIHpXOIxLop6l2quoUsMdR6OAgqoIrqHBH1/YucFpvO/zKKBfsv7Cr4zfSIfi52fBmuhKacsjH3l1XwNKgn9c2b6mCbb9J6W93Y/sOCdLC93b/q05JH6M17nJYEZA+uqFcDDc8L6YMmXcDbXG4r4+pChSEPGNZmgxWZlxYZx/YAnj/T0XBrmLhUiVUmTPN1xfUQh35TR4vTg5rBWnsF5RmOFvCrVllVwTeN5Ts8RU9ED12BYmWFkHmiK/po2om3N4s7RndpoTY5FyGS6oaXlt4exSebIT7Y2kpn/o26OJ/F/g4W+qJxX+qbbbHA5N6WJ2Q5W57/DqkBC+4aiXAzmt2rlcgrwLL87Tr4r0ZxUgzcJzsZ1ghT/bvr7C0UJaRPtl45n38eBwENdcMNnMjWQwOi4GyuPvQ2RrDVwwOikw6ztV0nqwIZE8k6zDC1MJRa+0roeghDi8SCdDce2wugqIRly9pWzFM+mhf01yLR6jIYYQztVY9tTYZCRdhzsX/ftxskA3NBi6BhEsgGUhwwyW4GUhwxSW4GkhwwSW4GEhwxiU4G0gg4hKIBhKccAlO/7k6HP9Xup/pO2esiBn8nHAJTgYSiLgEooEEZ1yCs4EEF1yCi4EEV1yCq4EEN1yCm4EEEi6BZCCBjEvQ+w9OKH1C2BGMg8h0ies8LCdEy9qrFSfqH3mEg2BixmpAVxaYvgkTTQ5TQm+FaWCtN+UgYx/iz4pBWEs0s78BmhqH9A=="],
    ImageSize -> Dynamic[{Automatic, 3.51 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}]
];


(* ::Subsection::Closed:: *)
(*Integer*)


(* ::Input:: *)
(*ArrayPlot[Round[2Rescale[Table[Sin[j^2+i],{i,Pi,0,-Pi/7},{j,0,Pi,Pi/7}]],1],Mesh->All,MeshStyle->Directive[AbsoluteThickness[0.1],GrayLevel[0.2]],ColorRules->{0->ColorData[112,2],1->ColorData[112,1],2->ColorData[112,3]},Frame->False]*)


$intgridicon = Show[
    Uncompress["1:eJztlc1Kw0AQx+MnrYjgI+itt+a7t0WQFqEgVC9ehLRu6mKskk0rPXkQ74IXW/AVvIsHhV7sE9jeBE+9+QjuJqHFNoMbSBDEhQyb2dlfZpf/ZDaqZxV7QZIkmmGm5Frnx6RG7XnuWWSmTKgXrC8zU7Goh13M3zPhw+dX17faZXmELp4Ou5svfSSFA/IH4xONSr0cpR8/+iFOr3jw3Ll7RdX6yv3bYIB2H3I3w+EIxeUklSfkh/KMm3/c+LTvOe69/eefbP5pc5Lip12nSfnT1knaef5W/lPt4tvbzhyb5AMbsVIILLhHjmpFs1+qNB1MV9lkq+ERyyEWJY06XWKOouVQHARmmdkmLq55pIVtTqfrfEuVnjlND++zznfSwJS63Q4fjyiI4dtYY2yXcQs74Vo/XAsyaWA7C98AeLaoU0feB0gwQYIpSDBAgiFI0EGCLkjQQIImSFBBgipIUECCIkiQQcK0huPXR/5P6T7qnJEVEREngwRZkKCABEWQoIIEVZCggQRNkKCDBF2QYIAEQ5BgggRTkFAACTP/QV/pvrBd6xRPS3wcseaLlhxxcdI9r+3giZgnNTAui4m+JX+8o1DoQ/QFB7hP1Q=="],
    ImageSize -> Dynamic[{Automatic, 3.51 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}]
];


(* ::Subsection::Closed:: *)
(*Vector*)


(* ::Input:: *)
(*VectorPlot[{-1-x^2+y,1+x-y^2},{x,-3,3},{y,-3,3},Frame->False,VectorPoints->4,VectorSizes->4{1,1},VectorStyle->{Arrowheads[0.22],Thickness[.03]}]*)


$vecgridicon = Graphics[
    {Arrowheads[0.2], Thickness[0.03],
    Uncompress["1:eJyV1XswXGcUAPBdj3omgla8aohWyEx34hGkIt9ukCYaGUFElDQ23lGSQzCtDBLZ6EwSKqp02kbbtBVDjEdEE9wNIR6LehVZFNtYQ8MKZcNa/XZGOpObP677z537zbn3u78559zzWXDjfML1GAxGvAq+HImKTwhXem2lrFipK1acA25xMXEA/oOdnvaW0wi8L8tawrIaEIiqOKPa6dMonKl4WBVfOABxSaSdXlvBOT8zuROSEvA2R2oXWy0nSPGWfgkjdEyND4k+6m7pjpp8Spcpenpxol2EoNY5YBpsxhE4KwvXHQkRLZe5QcGX1+VtBAynWB2f3i4ju/aNCI6yChcJyHGYt228okXtGjdOElrvxxzrqyvCIfkYgvmsjKaquj9ouVoME8JmXZYQTE2xj9l7rJFdafH1RUacWgTX5KJKs7lN5Mu4OjWwLGkSwc9fJK5+rvIUga6xjt5cwhQtF29qvrsjWoMNMs/vlVJbmaTvwiLLNCh0178IVKzFJ/e2qFG7AsRGQ/yTwwjYYx0ud9txvixKS5MiWU0EHZek16raxUZEwIcp25WzK2fI+VL5OFGvfI8yHzevNFByf5WgdJ29wcNF70GQWWbwqOo+di2xnI36jP+m5UoJ5ifkFc0iaPVaJALsJWTXRpwAp/ezv/v01EtqV8ux043C67h8U+sL3Ipa7KoWLOpo5/fQcu0eNLM1HFNmQ6a1qkCrYJnsYt/KcLmVijdP9kzNLpHOU7vyDD7RzcvFb/zVIdzRsYBv6h+NzeeXNtHqr9yfZsO4fvh/+3Nhv77RUCsixavvqnd09mvwX8WpXenF7jEB/p0IIvRHYsNrsOvHU8llgodiWvnSjo/iJoTiXTqTi0PU9z4m54vL8rp2Q7BKvIojSldDp8Axn4n7y8KmLYKv6C+/21ea6k/Q6y9WX60VowjPifNO5UU1o2Kyy7zOLnct9d7/cWrXr/ooq1FtAsFxoa1YpoUHWe9SX6VSwwCtOqpr5mXKtfGcmLMbKWS6jpJdkpneWaVJPCc24tSurSs6yILVhyCmuMuBo8jXu3tqQirrx2nla8huRSe4cpqAtH2T0cNBDDYp7hnPrLX2ZfKBsRi3Zb1snNo1PiCZ4Fi1IChRTrtg8jt2qVrz3DxiZ2i56pDj7FtXhwgA5wOziQ+YZNdGHME5/PFd8YPUrraczJnD73UjSJfHyl4o8nXGp83E/vQzWq72ZCco4OI50eMo8z3o/pz8P/aa7nD69gLulm9O8Oy+nlvZRH+5HvIvzHyG4E71kmrdwVEEsmZJec6cmFZ/TZ0xyTh8e40A6XTXEwdzbXK+tI++cBV+oMqHbduibNREm3A1q7SaD0TgducdMt/5FRPf6MbcjH7HtoeWK1JkVSFSf05A11lf78CyN1z5pkNepssTBHC3cg1rH7+kdnEjY90b8/H5eL5kuPlhB67jPe9fCtdkDbTqqHHzQUHOpXHFaQYhR354w7Uc4xH+2UV8Tkl4lksDk5twXQ6+tLCzX4hgS13CR0FPsKv7zj+7he2ttFzhhZZKdYVqbMhInagITdYkuxyImtUr9VIE0kk339/s5Og/Yy52Jw=="]},
    ImageSize -> Dynamic[{Automatic, 3.5 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}],
    PlotRange->{{-3,3},{-3,3}},
    PlotRangePadding->Scaled[.05]
];


(* ::Subsection::Closed:: *)
(*Boolean*)


(* ::Input:: *)
(*SeedRandom[610];*)
(*ArrayPlot[UnitStep[RandomReal[{0,1},{8,8}]-0.5],Mesh->All,MeshStyle->Directive[AbsoluteThickness[0.1],GrayLevel[0.2]],ColorRules->{0->Lighter[ColorData[112,4],0.25],1->Lighter[ColorData[112,1],0.25]},Frame->False]*)


$boolgridicon = Show[
    Uncompress["1:eJxTTMoPSmNmYGAo5gAS7kWJBRmZycVpTCARFiDhk1lcApFnAxJBicUlqUWpID4HFIPYSkvy0uN/vrL3aXv8k2fFTXsGMLhgj0scRksdnrPA1++ZvVZ/Xour61WS1ZNqDq3FSXUnrf1Fa/cMFX8NVLoabOFDLfVDJb0NlfJksIUPqeJo1QUKz5MRyDCAkFhkLCAkTj2G2KoiTJuCSnNSi3mADMe8kszEnMzE4sy89GJWoIBbYk5xKkQhJ5BwySxKTS7JLEtNA5leLAjSklScn1NakhoCrPmy81KLi4tmzQSBnfYQNSBtwIqx0ie1LDUHKncSKgdxSV5qGifuEMDpN2y+xhoeOE0wx2mCOZEmmOE0wYxIE0xxmmBKpAkmOE0wIdIEY5wmGBNpghFOE4yINMEQpwnoaZj0/GEwrNI9Nn9izRFY1BniNMGQSBOMcJpgRKQJxjhNMCbSBBOcJpgQaYIpThNMiTTBDKcJZkSaYI7TBHMiTbDAaQJGOQhO6eCEXZSYm4qexOEq+MCJNjMFlDiLg0sqc1IRiRmRB+DZApG+GcDggT00od+0BwB5P3G0"],
    ImageSize -> Dynamic[{Automatic, 3.51 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}]
];


(* ::Subsection::Closed:: *)
(*Mask*)


(* ::Input:: *)
(*SeedRandom[610];*)
(*ArrayPlot[UnitStep[RandomReal[{0,1},{8,8}]-0.5],Mesh->All,MeshStyle->Directive[AbsoluteThickness[0.1],GrayLevel[0.2]],ColorRules->{0->Black,1->None},Frame->False]*)


$maskgridicon = Show[
    Uncompress["1:eJzdlm1Kw0AQhuMnLYjgEbxB891/QRBF6K/aC6R1oosxlUxa6FH0Jh7BI/jPI3gEd7OhwXYHp0hI7UImH+/Mk93Nu2HPx9NhcmBZFnZkuM7j5wcxwWRfPTmUYSCw0PqxDMMYC8hB3XeqQyXp9hWZz1ydalTdpvV/5bet/9b/pvX/3v+m9bbnp21+299/2/8/2z5/Tevb7s9d/z7NjX9lu/Dj7mZPXvR0NCh9Hcka27QVWX/TcJYCnsiLi6wQcSpiFNk9HskHV3GKoBO7MlyKHCaFmEOi6HimSsY4TWcFjOTO5zEDxPz1RbW3SOeoMrkxWgxgDmmlvVea7kkGSZeeAXJsplEb54MkhCQhZBICkhAwCT5J8JkEjyR4TIJLElwmwSEJDpNgk4RVD2++Pno75XvTOI0rwpBnkwSbSXBIgsMkuCTBZRI8kuAxCT5J8JmEgCQETEJIEkImoU8S1v6DpdNLY+fxE6xafJlxWppW3Clz4m2xSKE2c70Glsui9rdVts+oMvpH9A2oYMqv"],
    ImageSize -> Dynamic[{Automatic, 3.51 Divide[CurrentValue["FontCapHeight"], AbsoluteCurrentValue[Magnification]]}]
];
