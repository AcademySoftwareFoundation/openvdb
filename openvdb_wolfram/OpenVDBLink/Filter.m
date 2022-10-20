(* ::Package:: *)

(* ::Title:: *)
(*Filter*)


(* ::Subtitle:: *)
(*Apply various filters to a level set or fog volume.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


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


OpenVDBFilter[args___] /; !CheckArgs[OpenVDBFilter[args], {2, 3}] = $Failed;


OpenVDBFilter[args___] :=
    With[{res = iOpenVDBFilter[args]},
        res /; res =!= $Failed
    ]


OpenVDBFilter[args___] := mOpenVDBFilter[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBFilter*)


iOpenVDBFilter[vdb_?OpenVDBScalarGridQ, filter_, iter_:1] /; levelSetQ[vdb] && IntegerQ[iter] && iter > 0 :=
    Block[{fdata, method, width},
        fdata = filteringMethod[filter];
        (
            {method, width} = fdata;

            vdb["filterGrid"[method, width, iter]];

            vdb

        ) /; fdata =!= $Failed
    ]


iOpenVDBFilter[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBFilter, 1];


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


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBFilter[expr_, ___] /; messageScalarGridQ[expr, OpenVDBFilter] = $Failed;


mOpenVDBFilter[expr_, ___] /; messageLevelSetGridQ[expr, OpenVDBFilter] = $Failed;


mOpenVDBFilter[_, filter_, ___] /; filteringMethod[filter] === $Failed :=
    (
        Message[OpenVDBFilter::filter, filter];
        $Failed
    )


mOpenVDBFilter[vdb_, filter_, expr_, rest___] /; !IntegerQ[expr] || !TrueQ[expr > 0] :=
    (
        Message[OpenVDBFilter::intpm, HoldForm[OpenVDBFillWithBalls[vdb, filter, expr, rest]], 3];
        $Failed
    )


mOpenVDBFilter[___] = $Failed;


OpenVDBFilter::filter = "`1` is not a valid filter.";
