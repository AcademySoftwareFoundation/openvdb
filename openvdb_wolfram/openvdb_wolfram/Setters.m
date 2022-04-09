(* ::Package:: *)

(* ::Title:: *)
(*Setters*)


(* ::Subtitle:: *)
(*Set various properties about a grid.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBSetProperty"]


OpenVDBSetProperty::usage = "OpenVDBSetProperty[expr, \"prop\", val] sets the value of property \"prop\" to val for the given OpenVDB grid.";


(* ::Section:: *)
(*OpenVDBSetProperty*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBSetProperty[vdb_?OpenVDBGridQ, args__] :=
	Block[{parsedargs, res},
		parsedargs = setterArguments[args];
		(
			res = MapThread[setVDBProperty[vdb, ##]&, parsedargs];
			
			If[MatchQ[{args}, {_String, _} | {_String -> _}],
				res[[1]],
				res
			]
			
		) /; parsedargs =!= $Failed
	]


OpenVDBSetProperty[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetProperty] = {"ArgumentsPattern" -> {_, _, _.}};


(* ::Section:: *)
(*Setters*)


(* ::Subsection::Closed:: *)
(*openVDBSetBackgroundValue*)


openVDBSetBackgroundValue[vdb_?OpenVDBBooleanGridQ, bool_?BooleanQ] := openVDBSetBackgroundValue[vdb, Boole[bool]]


openVDBSetBackgroundValue[vdb_?nonMaskGridQ, bg_] :=
	With[{res = Quiet[Check[vdb["setBackgroundValue"[bg]];True, False]]},
		Which[
			OpenVDBBooleanGridQ[vdb],
				Unitize[bg],
			Precision[vdb["getBackgroundValue"[]]] =!= \[Infinity],
				N[bg],
			True,
				bg
		] /; res
	]


openVDBSetBackgroundValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetCreator*)


openVDBSetCreator[vdb_, creator_String] :=
	(
		vdb["setCreator"[creator]];
		creator
	)


openVDBSetCreator[vdb_, None] := openVDBSetCreator[vdb, ""]


openVDBSetCreator[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetDescription*)


openVDBSetDescription[vdb_, description_String] :=
	(
		vdb["setStringMetadata"["description", description]];
		description
	)


openVDBSetDescription[vdb_, None] := openVDBSetDescription[vdb, ""]


openVDBSetDescription[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetGridClass*)


openVDBSetGridClass[vdb_?OpenVDBScalarGridQ, gc_] :=
	With[{gcid = gridClassID[gc]},
		vdb["setGridClass"[gcid]];
		gridClassName[gcid]
	]


openVDBSetGridClass[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetName*)


openVDBSetName[vdb_, name_String] :=
	(
		vdb["setName"[name]];
		name
	)


openVDBSetName[vdb_, None] := openVDBSetName[vdb, ""]


openVDBSetName[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetVoxelSize*)


openVDBSetVoxelSize[vdb_, spacing_?Positive] :=
	(
		vdb["setVoxelSize"[spacing]];
		N[spacing]	
	)


openVDBSetVoxelSize[___] = $Failed;


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*setterArguments*)


setterArguments[prop_String, val_] := {{prop}, {val}}
setterArguments[props_List, vals_List] /; Length[props] == Length[vals] := {props, vals}
setterArguments[prop_String -> val_] := {{prop}, {val}}
setterArguments[props_List -> vals_List] /; Length[props] == Length[vals] := {props, vals}
setterArguments[data:{(_ -> _)..}] := {data[[All, 1]], data[[All, 2]]}
setterArguments[___] = $Failed;


(* ::Subsection::Closed:: *)
(*setterPropertyFunctions*)


$setterPropertyAssoc = <|
	"BackgroundValue" -> openVDBSetBackgroundValue,
	"Creator" -> openVDBSetCreator,
	"Description" -> openVDBSetDescription,
	"GridClass" -> openVDBSetGridClass,
	"Name" -> openVDBSetName,
	"VoxelSize" -> openVDBSetVoxelSize
|>;


setVDBProperty[id_, prop_, val_] := 
	With[{setter = Lookup[$setterPropertyAssoc, prop, $Failed]},
		setter[id, val] /; setter =!= $Failed
	]


setVDBProperty[__] = $Failed;
