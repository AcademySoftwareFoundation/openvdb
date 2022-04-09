(* ::Package:: *)

(* ::Title:: *)
(*Image*)


(* ::Subtitle:: *)
(*Image representation through slices, depth map, projection, and Image3D.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBImage3D"]
PackageExport["OpenVDBDepthImage"]
PackageExport["OpenVDBProjectionImage"]
PackageExport["OpenVDBSliceImage"]
PackageExport["OpenVDBDynamicSliceImage"]


OpenVDBImage3D::usage = "OpenVDBImage3D[expr] returns an Image3D representation of an OpenVDB grid.";
OpenVDBDepthImage::usage = "OpenVDBDepthImage[expr] returns a top view depth map image of an OpenVDB grid.";
OpenVDBProjectionImage::usage = "OpenVDBProjectionImage[expr] returns a top view projection image of an OpenVDB grid.";
OpenVDBSliceImage::usage = "OpenVDBSliceImage[expr, z] returns a slice image of an OpenVDB grid at height z.";
OpenVDBDynamicSliceImage::usage = "OpenVDBDynamicSliceImage[expr] returns a Dynamic z slice viewer of expr.";


(* ::Section:: *)
(*Image3D*)


(* ::Subsection::Closed:: *)
(*OpenVDBImage3D*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBImage3D] = {Resampling -> Automatic, "ScalingFactor" -> 1.0};


OpenVDBImage3D[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ, OptionsPattern[]] :=
	Block[{bdsindex, scaling, resampling, vdbscaled, im3d},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		scaling = parseImage3DScaling[OptionValue["ScalingFactor"], bdsindex];
		resampling = OptionValue[Resampling];
		(
			bdsindex = regimeConvert[vdb, scaling * bds, regime -> $indexregime];
			vdbscaled = scaleVDB[vdb, scaling, resampling];
			(
				im3d = vdbscaled["gridImage3D"[bdsindex]];
				
				im3d /; ImageQ[im3d]
				
			) /; OpenVDBGridQ[vdbscaled]
			
		) /; scaling > 0
	]


OpenVDBImage3D[vdb_?carefulPixelGridQ, opts:OptionsPattern[]] := OpenVDBImage3D[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


OpenVDBImage3D[vdb_?carefulPixelGridQ, Automatic, opts:OptionsPattern[]] := OpenVDBImage3D[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


OpenVDBImage3D[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := OpenVDBImage3D[vdb, bspec -> $indexregime, opts]


OpenVDBImage3D[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ, opts:OptionsPattern[]] := 
	Block[{bds2d},
		bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];
		
		OpenVDBImage3D[vdb, Append[bds2d, int] -> regime, opts] /; bounds2DQ[bds2d]
	]


OpenVDBImage3D[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBImage3D, 1];


SyntaxInformation[OpenVDBImage3D] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBImage3D] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


parseImage3DScaling[x_?Positive, _] := x


parseImage3DScaling[Automatic, bds_] := Min[1.0, N[CubeRoot[$targetImage3DSize/indexVoxelCount[bds]]]]


parseImage3DScaling[___] = $Failed;


indexVoxelCount[bds_] := Times @@ (Abs[Subtract @@@ bds] + 1)


$targetImage3DSize = 1500000000;


scaleVDB[vdb_, s_ /; s == 1.0, _] := vdb


scaleVDB[vdb_, s_, resamp_] := 
	Block[{vdbscaled},
		vdbscaled = OpenVDBTransform[vdb, s, Resampling -> resamp];
		
		vdbscaled /; OpenVDBGridQ[vdbscaled]
	]


scaleVDB[___] = $Failed;


(* ::Section:: *)
(*Depth*)


(* ::Subsection::Closed:: *)
(*OpenVDBDepthImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBDepthImage[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ, \[Gamma]_?Positive, mnmx_] /; Length[mnmx] == 2 && VectorQ[mnmx, NumericQ] :=
	Block[{bdsindex, im},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		
		im = vdb["depthMap"[bdsindex, \[Gamma], mnmx[[1]], mnmx[[2]]]];
			
		im /; ImageQ[im]
	]


OpenVDBDepthImage[vdb_] := OpenVDBDepthImage[vdb, Automatic]


OpenVDBDepthImage[vdb_?carefulPixelGridQ, Automatic, args___] := OpenVDBDepthImage[vdb, vdb["IndexBoundingBox"] -> $indexregime, args]


OpenVDBDepthImage[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := OpenVDBDepthImage[vdb, bspec -> $indexregime, opts]


OpenVDBDepthImage[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ, args___] := 
	Block[{bds2d},
		bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];
		
		OpenVDBDepthImage[vdb, Append[bds2d, int] -> regime, args] /; bounds2DQ[bds2d]
	]


OpenVDBDepthImage[vdb_, bdspec_] := OpenVDBDepthImage[vdb, bdspec, 0.5]


OpenVDBDepthImage[vdb_, bdspec_, \[Gamma]_] := OpenVDBDepthImage[vdb, bdspec, \[Gamma], {0.2, 1.0}]


OpenVDBDepthImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBDepthImage, 1];


SyntaxInformation[OpenVDBDepthImage] = {"ArgumentsPattern" -> {_, _., _., _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBDepthImage] = $indexregime;


(* ::Section:: *)
(*Projection*)


(* ::Subsection::Closed:: *)
(*OpenVDBProjectionImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBProjectionImage[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ] :=
	Block[{bdsindex, im},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		
		im = vdb["depthMap"[bdsindex, 1.0, 1.0, 1.0]];
			
		(
			If[fogVolumeQ[vdb],
				Image[im, "Byte"],
				Image[im, "Bit"]
			]
			
		) /; ImageQ[im]
	]


OpenVDBProjectionImage[vdb_] := OpenVDBProjectionImage[vdb, Automatic]


OpenVDBProjectionImage[vdb_?carefulPixelGridQ, Automatic] := OpenVDBProjectionImage[vdb, vdb["IndexBoundingBox"] -> $indexregime]


OpenVDBProjectionImage[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := OpenVDBProjectionImage[vdb, bspec -> $indexregime, opts]


OpenVDBProjectionImage[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ] := 
	Block[{bds2d},
		bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];
		
		OpenVDBProjectionImage[vdb, Append[bds2d, int] -> regime] /; bounds2DQ[bds2d]
	]


OpenVDBProjectionImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBProjectionImage, 1];


SyntaxInformation[OpenVDBProjectionImage] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBProjectionImage] = $indexregime;


(* ::Section:: *)
(*Slice*)


(* ::Subsection::Closed:: *)
(*OpenVDBSliceImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBSliceImage] = {"MirrorSlice" -> False};


OpenVDBSliceImage[vdb_?carefulPixelGridQ, z_?NumericQ -> regime_?regimeQ, bds_?bounds2DQ, OptionsPattern[]] :=
	Block[{mirrorQ, threadedQ, zindex, bdsindex, im},
		mirrorQ = TrueQ[OptionValue["MirrorSlice"]];
		threadedQ = True;
		
		zindex = regimeConvert[vdb, z, regime -> $indexregime];
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		
		im = vdb["gridSliceImage"[zindex, bdsindex, mirrorQ, threadedQ]];
		
		im /; ImageQ[im]
	]


OpenVDBSliceImage[vdb_, z_?NumericQ, args___] := OpenVDBSliceImage[vdb, z -> $indexregime, args]


OpenVDBSliceImage[vdb_?carefulPixelGridQ, z_ -> regime_, Automatic, opts___] := 
	Block[{bbox},
		bbox = If[worldRegimeQ[regime], 
			"WorldBoundingBox", 
			"IndexBoundingBox"
		];
		OpenVDBSliceImage[vdb, z -> regime, Most[vdb[bbox]], opts]
	]


OpenVDBSliceImage[vdb_, z_, opts:OptionsPattern[]] := OpenVDBSliceImage[vdb, z, Automatic, opts]


OpenVDBSliceImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBSliceImage, 1];


SyntaxInformation[OpenVDBSliceImage] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSliceImage] = $indexregime;


(* ::Subsection::Closed:: *)
(*OpenVDBDynamicSliceImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDynamicSliceImage] = {DisplayFunction -> Identity, ImageSize -> Automatic};


OpenVDBDynamicSliceImage[vdb_?carefulPixelGridQ, OptionsPattern[]] /; !emptyVDBQ[vdb] :=
	DynamicModule[{x1, x2, y1, y2, z1, z2, start, end, ox1, ox2, oy1, oy2, zs, func, imsz, pxmax},
		{x1, x2, y1, y2, z1, z2} = Flatten[vdb["IndexBoundingBox"]];
		{x1, x2, y1, y2} += {-3, 3, -3, 3};
		{ox1, ox2, oy1, oy2} = {x1, x2, y1, y2};
		zs = Round[(z1+z2)/2];
		
		func = OptionValue[DisplayFunction];
		imsz = OptionValue[ImageSize];
		
		pxmax = If[OpenVDBBooleanGridQ[vdb] || OpenVDBMaskGridQ[vdb], 1, 255];
		
		Manipulate[
			If[TrueQ[OpenVDBGridQ[vdb]],
				Graphics[
					EventHandler[
						{
							Raster[Image`InternalImageData[func[OpenVDBSliceImage[vdb, Round[k], {{ox1, ox2}, {oy1, oy2}}]]], Automatic, {0, pxmax}], 
							Dynamic[If[ListQ[start], {EdgeForm[Directive[AbsoluteThickness[1], Red]], FaceForm[], Rectangle[start, MousePosition["Graphics"]]}, {}]]
						},
						{
							"MouseDown" :> (
								start = Round[MousePosition["Graphics"]]
							),
							"MouseUp" :> (
								end = Round[MousePosition["Graphics"]];
								If[start == end, 
									{ox1, ox2, oy1, oy2} = {x1, x2, y1, y2}, 
									{ox1, ox2, oy1, oy2} = {ox1, ox1-1, oy2+1, oy2} + Flatten[Sort /@ ({1, -1}Transpose[{start, end}])]
								];
								start=.
							)
						}
					],
					Frame -> True,
					FrameTicks -> None,
					ImageMargins -> 0,
					ImagePadding -> 25,
					ImageSize -> imsz,
					PlotRangePadding -> None
				],
				""
			],
			{{k, zs, "slice"}, z1, z2, 1},
			FrameMargins -> 0
		]
	]


OpenVDBDynamicSliceImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBDynamicSliceImage] = {"ArgumentsPattern" -> {_, OptionsPattern[]}};
