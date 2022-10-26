(* ::Package:: *)

(* ::Title:: *)
(*Values*)


(* ::Subtitle:: *)
(*Set and retrieve states and values.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBSetStates"]
PackageExport["OpenVDBSetValues"]
PackageExport["OpenVDBStates"]
PackageExport["OpenVDBValues"]


OpenVDBSetStates::usage = "OpenVDBSetStates[expr, coords, states] sets the states at coordinates of a given OpenVDB expression.";
OpenVDBSetValues::usage = "OpenVDBSetValues[expr, coords, vals] sets the values at coordinates of a given OpenVDB expression.";


OpenVDBStates::usage = "OpenVDBStates[expr, coords] returns the state at coordinates of a given OpenVDB expression.";
OpenVDBValues::usage = "OpenVDBValues[expr, coords] returns the values at coordinates of a given OpenVDB expression.";


(* ::Section:: *)
(*States*)


(* ::Subsection::Closed:: *)
(*OpenVDBSetStates*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBSetStates[args___] /; !CheckArgs[OpenVDBSetStates[args], 3] = $Failed;


OpenVDBSetStates[args___] :=
    With[{res = iOpenVDBSetStates[args]},
        res /; res =!= $Failed
    ]


OpenVDBSetStates[args___] := mOpenVDBSetStates[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBSetStates*)


iOpenVDBSetStates[vdb_?OpenVDBGridQ, coords_?coordinatesQ -> regime_?regimeQ, states_] /; VectorQ[states, realQ] && Length[coords] === Length[states] :=
    With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
        vdb["setActiveStates"[indexcoordinates, states]];

        Unitize[states]
    ]


iOpenVDBSetStates[expr_, coord_?coordinateQ -> regime_, state_] /; realQ[state] || BooleanQ[state] :=
    With[{res = iOpenVDBSetStates[expr, {coord} -> regime, {state}]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetStates[expr_, coords_ -> regime_, states_List, args___] /; VectorQ[states, BooleanQ] := iOpenVDBSetStates[expr, coords -> regime, Boole[states], args]


iOpenVDBSetStates[expr_, coords_?MatrixQ -> regime_, state_] /; realQ[state] || BooleanQ[state] :=
    With[{res = iOpenVDBSetStates[expr, coords -> regime, ConstantArray[state, Length[coords]]]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetStates[expr_, coords_List, args___] := iOpenVDBSetStates[expr, coords -> $indexregime, args]


iOpenVDBSetStates[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetStates] = {"ArgumentsPattern" -> {_, _, _}};


OpenVDBDefaultSpace[OpenVDBSetStates] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBSetStates[expr_, ___] /; messageGridQ[expr, OpenVDBSetStates, False] = $Failed;


mOpenVDBSetStates[_, coords_, ___] /; messageCoordinateSpecQ[coords, OpenVDBSetStates] = $Failed;


mOpenVDBSetStates[___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBStates*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBStates[args___] /; !CheckArgs[OpenVDBStates[args], 2] = $Failed;


OpenVDBStates[args___] :=
    With[{res = iOpenVDBStates[args]},
        res /; res =!= $Failed
    ]


OpenVDBStates[args___] := mOpenVDBStates[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBStates*)


iOpenVDBStates[vdb_?OpenVDBGridQ, coords_?coordinatesQ -> regime_?regimeQ] :=
    With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
        vdb["getActiveStates"[indexcoordinates]]
    ]


iOpenVDBStates[expr_, coord_?coordinateQ -> regime_] :=
    With[{vals = iOpenVDBStates[expr, {coord} -> regime]},
        vals[[1]] /; vals =!= $Failed
    ]


iOpenVDBStates[expr_, coords_List] := iOpenVDBStates[expr, coords -> $indexregime]


iOpenVDBStates[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBStates] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBStates] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBStates[expr_, ___] /; messageGridQ[expr, OpenVDBStates, False] = $Failed;


mOpenVDBStates[_, coords_] /; messageCoordinateSpecQ[coords, OpenVDBStates] = $Failed;


mOpenVDBStates[___] = $Failed;


(* ::Section:: *)
(*Values*)


(* ::Subsection::Closed:: *)
(*OpenVDBSetValues*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBSetValues[args___] /; !CheckArgs[OpenVDBSetValues[args], 3] = $Failed;


OpenVDBSetValues[args___] :=
    With[{res = iOpenVDBSetValues[args]},
        res /; res =!= $Failed
    ]


OpenVDBSetValues[args___] := mOpenVDBSetValues[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBSetValues*)


iOpenVDBSetValues[vdb_?carefulNonMaskGridQ, coords_?coordinatesQ -> regime_?regimeQ, values_] /; ArrayQ[values, _, realQ] && Length[coords] === Length[values] :=
    Block[{indexcoordinates, res},
        indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime];

        res = Quiet[vdb["setValues"[indexcoordinates, values]]];

        Which[
            OpenVDBBooleanGridQ[vdb],
                Unitize[values],
            Precision[vdb["getBackgroundValue"[]]] =!= \[Infinity],
                N[values],
            True,
                values
        ] /; res === Null
    ]


iOpenVDBSetValues[expr_?OpenVDBVectorGridQ, coord_?coordinateQ -> regime_, value_?VectorQ] :=
    With[{res = iOpenVDBSetValues[expr, {coord} -> regime, {value}]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetValues[expr_?OpenVDBVectorGridQ, coords_?MatrixQ -> regime_, value_?VectorQ] :=
    With[{res = iOpenVDBSetValues[expr, coords -> regime, ConstantArray[value, Length[coords]]]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetValues[expr_, coord_?coordinateQ -> regime_, value_?realQ] :=
    With[{res = iOpenVDBSetValues[expr, {coord} -> regime, {value}]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetValues[expr_, coords_?MatrixQ -> regime_, value_?realQ] :=
    With[{res = iOpenVDBSetValues[expr, coords -> regime, ConstantArray[value, Length[coords]]]},
        res[[1]] /; res =!= $Failed
    ]


iOpenVDBSetValues[expr_, coords_List, args___] := iOpenVDBSetValues[expr, coords -> $indexregime, args]


iOpenVDBSetValues[vdb_?OpenVDBBooleanGridQ, coords_?coordinatesQ -> regime_?regimeQ, values:{(True|False)..}] :=
    iOpenVDBSetValues[vdb, coords -> regime, Boole[values]]


iOpenVDBSetValues[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetValues] = {"ArgumentsPattern" -> {_, _, _}};


OpenVDBDefaultSpace[OpenVDBSetValues] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBSetValues[expr_, ___] /; messageNonMaskGridQ[expr, OpenVDBSetValues] = $Failed;


mOpenVDBSetValues[_, coords_, ___] /; messageCoordinateSpecQ[coords, OpenVDBSetValues] = $Failed;


mOpenVDBSetValues[___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBValues*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBValues[args___] /; !CheckArgs[OpenVDBValues[args], 2] = $Failed;


OpenVDBValues[args___] :=
    With[{res = iOpenVDBValues[args]},
        res /; res =!= $Failed
    ]


OpenVDBValues[args___] := mOpenVDBValues[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBValues*)


iOpenVDBValues[vdb_?carefulNonMaskGridQ, coords_?coordinatesQ -> regime_?regimeQ] :=
    With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
        vdb["getValues"[indexcoordinates]]
    ]


iOpenVDBValues[expr_, coord_?coordinateQ -> regime_] :=
    With[{vals = iOpenVDBValues[expr, {coord} -> regime]},
        vals[[1]] /; vals =!= $Failed
    ]


iOpenVDBValues[expr_, coords_List] := iOpenVDBValues[expr, coords -> $indexregime]


iOpenVDBValues[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBValues] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBValues] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBValues[expr_, ___] /; messageNonMaskGridQ[expr, OpenVDBValues] = $Failed;


mOpenVDBValues[_, coords_] /; messageCoordinateSpecQ[coords, OpenVDBValues] = $Failed;


mOpenVDBValues[___] = $Failed;
