(* ::Package:: *)

(* ::Title:: *)
(*Transform*)


(* ::Subtitle:: *)
(*Transform level set indices or values.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBTransform"]
PackageExport["OpenVDBMultiply"]
PackageExport["OpenVDBGammaAdjust"]


OpenVDBTransform::usage = "OpenVDBTransform[expr, tfunc] applies a geometric transform tfunc on an OpenVDB grid.";


OpenVDBMultiply::usage = "OpenVDBMultiply[expr, s] multiplies all values by s, and for fog volumes, clipping values to lie between 0 and 1.";
OpenVDBGammaAdjust::usage = "OpenVDBGammaAdjust[expr, \[Gamma]] applies gamma adjustment on a fog volume.";


(* ::Section:: *)
(*Geometric transformation*)


(* ::Subsection::Closed:: *)
(*OpenVDBTransform*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBTransform] = {Resampling -> Automatic};


OpenVDBTransform[args___] /; !CheckArgs[OpenVDBTransform[args], 2] = $Failed;


OpenVDBTransform[args___] :=
    With[{res = iOpenVDBTransform[args]},
        res /; res =!= $Failed
    ]


OpenVDBTransform[args___] := mOpenVDBTransform[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBTransform*)


Options[iOpenVDBTransform] = Options[OpenVDBTransform];


iOpenVDBTransform[vdb_?OpenVDBGridQ, tf_, OptionsPattern[]] :=
    Block[{ptf, mat, regime, vx, tvdb, resampling, tfunc},
        ptf = ParseTransformationMatrix[tf];
        resampling = resamplingMethod[OptionValue[Resampling]];
        (
            {mat, regime} = ptf;
            vx = voxelSize[vdb];
            tvdb = OpenVDBCreateGrid[vdb];

            tfunc = Transpose[mat];
            If[worldRegimeQ[regime],
                tfunc[[1 ;; 3, 4]] *= vx;
                tfunc[[4, 1 ;; 3]] /= vx;
            ];

            tvdb["transformGrid"[vdb[[1]], tfunc, resampling]];

            tvdb

        ) /; ListQ[ptf] && Length[ptf] === 2 && resampling =!= $Failed
    ]


iOpenVDBTransform[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBTransform, 1];


SyntaxInformation[OpenVDBTransform] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBTransform] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


ParseTransformationMatrix[tf_] :=
    Catch[
        iParseTransformationMatrix @ If[Head[tf] === Rule,
            tf,
            tf -> $worldregime
        ]
    ]


iParseTransformationMatrix[tf_ -> regime_?regimeQ] := {iParseTransformationMatrix[tf], regime}


iParseTransformationMatrix[mat_List] /; MatrixQ[mat, realQ] && Dimensions[mat] === {4, 4} := mat


iParseTransformationMatrix[tfunc_TransformationFunction] := iParseTransformationMatrix[tfunc["TransformationMatrix"]]


iParseTransformationMatrix[mat_List] /; MatrixQ[mat, realQ] && Dimensions[mat] === {3, 3} := iParseTransformationMatrix[AffineTransform[mat]]


iParseTransformationMatrix[vec_List] /; VectorQ[vec, realQ] && Length[vec] == 3 := iParseTransformationMatrix[TranslationTransform[vec]]


iParseTransformationMatrix[scale_?Positive] := iParseTransformationMatrix[ScalingTransform[scale{1,1,1}]]


iParseTransformationMatrix[___] := Throw[$Failed]


(* ::Subsubsection::Closed:: *)
(*Messages*)


Options[mOpenVDBTransform] = Options[OpenVDBTransform];


mOpenVDBTransform[expr_, ___] /; messageGridQ[expr, OpenVDBTransform] = $Failed;


mOpenVDBTransform[_, tf_, ___] /; ParseTransformationMatrix[tf] === $Failed :=
    (
        Message[OpenVDBTransform::trans, tf, 2];
        $Failed
    )


mOpenVDBTransform[__, OptionsPattern[]] :=
    Block[{resampling},
        resampling = resamplingMethod[OptionValue[Resampling]];
        (
            Message[OpenVDBTransform::resamp];
            $Failed
        ) /; resampling === $Failed
    ]


mOpenVDBTransform[___] = $Failed;


OpenVDBTransform::trans = "`1` at position `2` does not represent a valid geometric transformation.";


OpenVDBTransform::resamp = "The setting for Resampling should be one of \"Nearest\", \"Linear\", \"Quadratic\", or Automatic.";


(* ::Section:: *)
(*Value transformation*)


(* ::Subsection::Closed:: *)
(*OpenVDBMultiply*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBMultiply[args___] /; !CheckArgs[OpenVDBMultiply[args], 2] = $Failed;


OpenVDBMultiply[args___] :=
    With[{res = iOpenVDBMultiply[args]},
        res /; res =!= $Failed
    ]


OpenVDBMultiply[args___] := mOpenVDBMultiply[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMultiply*)


iOpenVDBMultiply[vdb_?OpenVDBScalarGridQ, s_?realQ] :=
    (
        If[s != 1.0,
            vdb["scalarMultiply"[s]]
        ];
        vdb
    )


iOpenVDBMultiply[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMultiply, 1];


SyntaxInformation[OpenVDBMultiply] = {"ArgumentsPattern" -> {_, _}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBMultiply[expr_, ___] /; messageScalarGridQ[expr, OpenVDBMultiply] = $Failed;


mOpenVDBMultiply[_, s_] /; !realQ[s] :=
    (
        Message[OpenVDBMultiply::real, s, 2];
        $Failed
    )


mOpenVDBMultiply[___] = $Failed;


OpenVDBMultiply::real = "`1` at position `2` is not real number.";


(* ::Subsection::Closed:: *)
(*OpenVDBGammaAdjust*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGammaAdjust[args___] /; !CheckArgs[OpenVDBGammaAdjust[args], 2] = $Failed;


OpenVDBGammaAdjust[args___] :=
    With[{res = iOpenVDBGammaAdjust[args]},
        res /; res =!= $Failed
    ]


OpenVDBGammaAdjust[args___] := mOpenVDBGammaAdjust[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBGammaAdjust*)


iOpenVDBGammaAdjust[vdb_?OpenVDBScalarGridQ, \[Gamma]_?Positive] /; fogVolumeQ[vdb] :=
    (
        If[\[Gamma] != 1.0,
            vdb["gammaAdjustment"[\[Gamma]]]
        ];
        vdb
    )


iOpenVDBGammaAdjust[vdb_?OpenVDBScalarGridQ, args___] /; levelSetQ[vdb] :=
    (
        OpenVDBToFogVolume[vdb];
        iOpenVDBGammaAdjust[vdb, args]
    )


iOpenVDBGammaAdjust[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBGammaAdjust, 1];


SyntaxInformation[OpenVDBGammaAdjust] = {"ArgumentsPattern" -> {_, _}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBGammaAdjust[expr_, ___] /; messageScalarGridQ[expr, OpenVDBGammaAdjust] = $Failed;


mOpenVDBGammaAdjust[_, \[Gamma]_] /; !TrueQ[\[Gamma] > 0] :=
    (
        Message[OpenVDBGammaAdjust::pos, \[Gamma], 2];
        $Failed
    )


mOpenVDBGammaAdjust[___] = $Failed;


OpenVDBGammaAdjust::pos = "`1` at position `2` is not a positive number.";
