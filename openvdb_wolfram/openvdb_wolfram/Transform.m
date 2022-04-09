(* ::Package:: *)

(* ::Title:: *)
(*Transform*)


(* ::Subtitle:: *)
(*Transform level set indices or values.*)


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


OpenVDBTransform[vdb_?OpenVDBGridQ, tf_, OptionsPattern[]] :=
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


OpenVDBTransform[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBTransform, 1];


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


(* ::Section:: *)
(*Value transformation*)


(* ::Subsection::Closed:: *)
(*OpenVDBMultiply*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBMultiply[vdb_?OpenVDBScalarGridQ, s_?realQ] := 
	(
		If[s != 1.0, 
			vdb["scalarMultiply"[s]]
		];
		vdb
	)


OpenVDBMultiply[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBMultiply, 1];


SyntaxInformation[OpenVDBMultiply] = {"ArgumentsPattern" -> {_, _}};


(* ::Subsection::Closed:: *)
(*OpenVDBGammaAdjust*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBGammaAdjust[vdb_?OpenVDBScalarGridQ, \[Gamma]_?Positive] /; fogVolumeQ[vdb] := 
	(
		If[\[Gamma] != 1.0, 
			vdb["gammaAdjustment"[\[Gamma]]]
		];
		vdb
	)


OpenVDBGammaAdjust[vdb_?OpenVDBScalarGridQ, args___] /; levelSetQ[vdb] :=
	(
		OpenVDBToFogVolume[vdb];
		OpenVDBGammaAdjust[vdb, args]
	)


OpenVDBGammaAdjust[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBGammaAdjust, 1];


SyntaxInformation[OpenVDBGammaAdjust] = {"ArgumentsPattern" -> {_, _}};
