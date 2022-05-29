(* ::Package:: *)

(* ::Title:: *)
(*Messages*)


(* ::Subtitle:: *)
(*General message utilities*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageScope["messageGridQ"]
PackageScope["messageScalarGridQ"]
PackageScope["messageNonMaskGridQ"]


PackageScope["messageCoordinateSpecQ"]
PackageScope["messagRegimeSpecQ"]


PackageScope["messageZSpecQ"]
PackageScope["messageZSliceQ"]
PackageScope["message2DBBoxQ"]
PackageScope["message3DBBoxQ"]


PackageScope["messageIsoValueQ"]


(* ::Section:: *)
(*Grid*)


(* ::Subsection::Closed:: *)
(*messageGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageGridQ[_?OpenVDBGridQ, _] = False;


messageGridQ[expr_, head_] :=
	Block[{regionQ},
		regionQ = ConstantRegionQ[expr] && RegionEmbeddingDimension[expr] === 3;
		Which[
			TrueQ[$OpenVDBSpacing > 0] && regionQ,
				False, 
			!TrueQ[$OpenVDBSpacing > 0],
				Message[head::grid, expr];
				True, 
			True,
				Message[head::grid2, expr];
				True
		]
	]


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::grid = "`1` is not a grid.";
General::grid2 = "`1` is not a grid or constant 3D region.";


(* ::Subsection::Closed:: *)
(*messageScalarGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageScalarGridQ[_?OpenVDBScalarGridQ, _] = False;


messageScalarGridQ[expr_, head_] :=
	Block[{regionQ},
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


(* ::Subsection::Closed:: *)
(*messageNonMaskGridQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageNonMaskGridQ[expr_?OpenVDBGridQ, head_] :=
	If[!carefulNonMaskGridQ[expr],
		Message[head::nmsksupp, head];
		True,
		False
	]


messageNonMaskGridQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::nmsksupp = "`1` does not support mask grids.";


(* ::Section:: *)
(*Coordinates*)


(* ::Subsection::Closed:: *)
(*messageCoordinateSpecQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageCoordinateSpecQ[expr_ -> regime_, head_] := messagRegimeSpecQ[regime, head] || messageCoordinateSpecQ[expr, head];


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
(*messagRegimeSpecQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messagRegimeSpecQ[_?regimeQ, _] = False;


messagRegimeSpecQ[expr_, head_] := (Message[head::gridspace, expr];True)


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::gridspace = "`1` is not one of \"Index\" or \"World\".";


(* ::Section:: *)
(*Bounds*)


(* ::Subsection::Closed:: *)
(*messageZSpecQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageZSpecQ[zspec_List -> regime_, head_] := messagRegimeSpecQ[regime, head] || messageZSpecQ[zspec, head]


messageZSpecQ[zspec_, head_] := 
	If[!MatchQ[zspec, Automatic|({z1_, z2_} /; z1 <= z2)],
		Message[head::zspec, zspec];
		True,
		False
	]


messageZSpecQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::zspec = "`1` does not represent valid z\[Hyphen]bounds.";


(* ::Subsection::Closed:: *)
(*messageZSliceQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


messageZSliceQ[z_List -> regime_, head_] := messagRegimeSpecQ[regime, head] || messageZSliceQ[z, head]


messageZSliceQ[z_, head_] := 
	If[!realQ[z],
		Message[head::zslice, z];
		True,
		False
	]


messageZSliceQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::zslice = "`1` does not represent a valid z\[Hyphen]slice.";


(* ::Subsection::Closed:: *)
(*message2DBBoxQ*)


(* ::Subsubsection:: *)
(*Main*)


message2DBBoxQ[bbox_List -> regime_, head_] := messagRegimeSpecQ[regime, head] || message2DBBoxQ[bbox, head]


message2DBBoxQ[bbox_, head_] := 
	If[!bounds2DQ[bbox] && bbox =!= Automatic,
		Message[head::bbox2d, bbox];
		True,
		False
	]


message2DBBoxQ[___] = False;


(* ::Subsubsection:: *)
(*Messages*)


General::bbox2d = "`1` does not represent valid 2D bunding box.";


(* ::Subsection::Closed:: *)
(*message3DBBoxQ*)


(* ::Subsubsection::Closed:: *)
(*Main*)


message3DBBoxQ[bbox_List -> regime_, head_] := messagRegimeSpecQ[regime, head] || message3DBBoxQ[bbox, head]


message3DBBoxQ[bbox_, head_] := 
	If[!bounds3DQ[bbox] && bbox =!= Automatic,
		Message[head::bbox3d, bbox];
		True,
		False
	]


message3DBBoxQ[___] = False;


(* ::Subsubsection::Closed:: *)
(*Messages*)


General::bbox3d = "`1` does not represent valid 3D bunding box.";


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
