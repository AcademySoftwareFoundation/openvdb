(* ::Package:: *)

(* ::Title:: *)
(*Messages*)


(* ::Subtitle:: *)
(*General message utilities*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["messageScalarGridQ"]


PackageExport["messageCoordinateSpecQ"]
PackageExport["invalidRegimeSpecQ"]


PackageExport["messageIsoValueQ"]


(* ::Section:: *)
(*Grid*)


(* ::Subsection::Closed:: *)
(*messageScalarGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageScalarGridQ[_?OpenVDBScalarGridQ, _] = False;


messageScalarGridQ[expr_, head_] :=
	Block[{scalarGridQ, regionQ},
		regionQ = ConstantRegionQ[expr] && RegionEmbeddingDimension[expr] === 3;
		Which[
			TrueQ[$OpenVDBSpacing > 0] && regionQ,
				False, 
			!TrueQ[$OpenVDBSpacing > 0],
				Message[head::scalargrid, expr];
				True, 
			True,
				Message[head::scalargrid2, expr];
				True
		]
	]


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::scalargrid = "`1` is not a scalar grid.";
General::scalargrid2 = "`1` is not a scalar grid or constant 3D region.";


(* ::Section:: *)
(*Coordinates*)


(* ::Subsection::Closed:: *)
(*messageCoordinateSpecQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageCoordinateSpecQ[expr_ -> regime_, head_] := invalidRegimeSpecQ[regime, head] || messageCoordinateSpecQ[expr, head];


messageCoordinateSpecQ[expr_, head_] :=
	If[coordinateQ[expr] || coordinatesQ[expr],
		False,
		Message[head::coord, expr];
		True
	]


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::coord = "`1` is not a 3D coordinate or collection of 3D coordinates.";


(* ::Subsection::Closed:: *)
(*invalidRegimeSpecQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


invalidRegimeSpecQ[_?regimeQ, _] = False;


invalidRegimeSpecQ[expr_, head_] := (Message[head::gridspace, expr];True)


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::gridspace = "`1` is not one of \"Index\" or \"World\".";


(* ::Section:: *)
(*Common options*)


(* ::Subsection::Closed:: *)
(*messageIsoValueQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageIsoValueQ[iso_, head_] := 
	If[!realQ[iso] && iso =!= Automatic,
		Message[head::isoval];
		True,
		False
	]


messageIsoValueQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::isoval = "The setting for \"IsoValue\" should either be a real number or Automatic.";
