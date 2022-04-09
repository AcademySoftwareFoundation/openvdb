(* ::Package:: *)

(* ::Title:: *)
(*Values*)


(* ::Subtitle:: *)
(*Set and retrieve states and values.*)


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


OpenVDBSetStates[vdb_?OpenVDBGridQ, coords_?coordinatesQ -> regime_?regimeQ, states_] /; VectorQ[states, realQ] && Length[coords] === Length[states] :=
	With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
		vdb["setActiveStates"[indexcoordinates, states]];
		
		Unitize[states]
	]


OpenVDBSetStates[expr_, coord_?coordinateQ -> regime_, state_] /; realQ[state] || BooleanQ[state] :=
	With[{res = OpenVDBSetStates[expr, {coord} -> regime, {state}]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetStates[expr_, coords_ -> regime_, states_List, args___] /; VectorQ[states, BooleanQ] := OpenVDBSetStates[expr, coords -> regime, Boole[states], args]


OpenVDBSetStates[expr_, coords_?MatrixQ -> regime_, state_] /; realQ[state] || BooleanQ[state] :=
	With[{res = OpenVDBSetStates[expr, coords -> regime, ConstantArray[state, Length[coords]]]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetStates[expr_, coords_List, args___] := OpenVDBSetStates[expr, coords -> $indexregime, args]


OpenVDBSetStates[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetStates] = {"ArgumentsPattern" -> {_, _, _}};


OpenVDBDefaultSpace[OpenVDBSetStates] = $indexregime;


(* ::Subsection::Closed:: *)
(*OpenVDBStates*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBStates[vdb_?OpenVDBGridQ, coords_?coordinatesQ -> regime_?regimeQ] :=
	With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
		vdb["getActiveStates"[indexcoordinates]]
	]


OpenVDBStates[expr_, coord_?coordinateQ -> regime_] :=
	With[{vals = OpenVDBStates[expr, {coord} -> regime]},
		vals[[1]] /; vals =!= $Failed
	]


OpenVDBStates[expr_, coords_List] := OpenVDBStates[expr, coords -> $indexregime]


OpenVDBStates[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBStates] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBStates] = $indexregime;


(* ::Section:: *)
(*Values*)


(* ::Subsection::Closed:: *)
(*OpenVDBSetValues*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBSetValues[vdb_?carefulNonMaskGridQ, coords_?coordinatesQ -> regime_?regimeQ, values_] /; ArrayQ[values, _, realQ] && Length[coords] === Length[values] :=
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


OpenVDBSetValues[expr_?OpenVDBVectorGridQ, coord_?coordinateQ -> regime_, value_?VectorQ] :=
	With[{res = OpenVDBSetValues[expr, {coord} -> regime, {value}]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetValues[expr_?OpenVDBVectorGridQ, coords_?MatrixQ -> regime_, value_?VectorQ] :=
	With[{res = OpenVDBSetValues[expr, coords -> regime, ConstantArray[value, Length[coords]]]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetValues[expr_, coord_?coordinateQ -> regime_, value_?realQ] :=
	With[{res = OpenVDBSetValues[expr, {coord} -> regime, {value}]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetValues[expr_, coords_?MatrixQ -> regime_, value_?realQ] :=
	With[{res = OpenVDBSetValues[expr, coords -> regime, ConstantArray[value, Length[coords]]]},
		res[[1]] /; res =!= $Failed
	]


OpenVDBSetValues[expr_, coords_List, args___] := OpenVDBSetValues[expr, coords -> $indexregime, args]


OpenVDBSetValues[vdb_?OpenVDBBooleanGridQ, coords_?coordinatesQ -> regime_?regimeQ, values:{(True|False)..}] :=
	OpenVDBSetValues[vdb, coords -> regime, Boole[values]]


OpenVDBSetValues[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetValues] = {"ArgumentsPattern" -> {_, _, _}};


OpenVDBDefaultSpace[OpenVDBSetValues] = $indexregime;


(* ::Subsection::Closed:: *)
(*OpenVDBValues*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBValues[vdb_?carefulNonMaskGridQ, coords_?coordinatesQ -> regime_?regimeQ] :=
	With[{indexcoordinates = regimeConvert[vdb, coords, regime -> $indexregime]},
		vdb["getValues"[indexcoordinates]]
	]


OpenVDBValues[expr_, coord_?coordinateQ -> regime_] :=
	With[{vals = OpenVDBValues[expr, {coord} -> regime]},
		vals[[1]] /; vals =!= $Failed
	]


OpenVDBValues[expr_, coords_List] := OpenVDBValues[expr, coords -> $indexregime]


OpenVDBValues[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBValues] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBValues] = $indexregime;
