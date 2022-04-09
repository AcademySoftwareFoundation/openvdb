(* ::Package:: *)

(* ::Title:: *)
(*Filter*)


(* ::Subtitle:: *)
(*Apply various filters to a level set or fog volume.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBFilter"]


OpenVDBFilter::usage = "OpenVDBFilter[expr, f] applies a filter on level set OpenVDB grid.";


(* ::Section:: *)
(*Filtering*)


(* ::Subsection::Closed:: *)
(*OpenVDBFilter*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBFilter[vdb_?OpenVDBScalarGridQ, filter_, iter_:1] /; levelSetQ[vdb] && IntegerQ[iter] && iter > 0 :=
	Block[{fdata, method, width},
		fdata = filteringMethod[filter];
		(
			{method, width} = fdata;
			
			vdb["filterGrid"[method, width, iter]];
			
			vdb
			
		) /; fdata =!= $Failed
	]


OpenVDBFilter[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBFilter, 1];


SyntaxInformation[OpenVDBFilter] = {"ArgumentsPattern" -> {_, _, _.}};


addCodeCompletion[OpenVDBFilter][None, {"Mean", "Median", "Gaussian", "Laplacian", "MeanCurvature"}, None];


(* ::Subsubsection::Closed:: *)
(*Utilities*)


filteringMethod[{"Mean", r_Integer?Positive}] := {0, r}
filteringMethod[{"Median", r_Integer?Positive}] := {1, r}
filteringMethod[{"Gaussian", r_Integer?Positive}] := {2, r}
filteringMethod["Mean"] := {0, 1}
filteringMethod["Median"] := {1, 1}
filteringMethod["Gaussian"] := {2, 1}
filteringMethod["Laplacian"] := {3, 1}
filteringMethod["MeanCurvature"] := {4, 1}
filteringMethod[___] = $Failed;
