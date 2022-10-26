(* ::Package:: *)

(* ::Title:: *)
(*Image*)


(* ::Subtitle:: *)
(*Image representation through slices, depth map, projection, and Image3D.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


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


OpenVDBImage3D[args___] /; !CheckArgs[OpenVDBImage3D[args], {1, 2}] = $Failed;


OpenVDBImage3D[args___] :=
    With[{res = iOpenVDBImage3D[args]},
        res /; res =!= $Failed
    ]


OpenVDBImage3D[args___] := mOpenVDBImage3D[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBImage3D*)


Options[iOpenVDBImage3D] = Options[OpenVDBImage3D];


iOpenVDBImage3D[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ, OptionsPattern[]] :=
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


iOpenVDBImage3D[vdb_?carefulPixelGridQ, opts:OptionsPattern[]] := iOpenVDBImage3D[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


iOpenVDBImage3D[vdb_?carefulPixelGridQ, Automatic, opts:OptionsPattern[]] := iOpenVDBImage3D[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


iOpenVDBImage3D[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := iOpenVDBImage3D[vdb, bspec -> $indexregime, opts]


iOpenVDBImage3D[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ, opts:OptionsPattern[]] :=
    Block[{bds2d},
        bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];

        iOpenVDBImage3D[vdb, Append[bds2d, int] -> regime, opts] /; bounds2DQ[bds2d]
    ]


iOpenVDBImage3D[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBImage3D, 1];


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


(* ::Subsubsection::Closed:: *)
(*Messages*)


Options[mOpenVDBImage3D] = Options[OpenVDBImage3D];


mOpenVDBImage3D[expr_, ___] /; messageGridQ[expr, OpenVDBImage3D] = $Failed;


mOpenVDBImage3D[expr_, ___] /; messagePixelGridQ[expr, OpenVDBImage3D] = $Failed;


mOpenVDBImage3D[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBImage3D] = $Failed;


mOpenVDBImage3D[__, OptionsPattern[]] :=
    Block[{s},
        s = parseImage3DScaling[OptionValue["ScalingFactor"], {{0,1},{0,1},{0,1}}];
        (
            Message[OpenVDBImage3D::scale];
            $Failed
        ) /; s === $Failed
    ]


mOpenVDBImage3D[___] = $Failed;


OpenVDBImage3D::scale = "The setting for \"ScalingFactor\" should either be a positive number or Automatic.";


(* ::Section:: *)
(*Depth*)


(* ::Subsection::Closed:: *)
(*OpenVDBDepthImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBDepthImage[args___] /; !CheckArgs[OpenVDBDepthImage[args], {1, 4}] = $Failed;


OpenVDBDepthImage[args___] :=
    With[{res = iOpenVDBDepthImage[args]},
        res /; res =!= $Failed
    ]


OpenVDBDepthImage[args___] := mOpenVDBDepthImage[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDepthImage*)


iOpenVDBDepthImage[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ, \[Gamma]_?Positive, mnmx_] /; Length[mnmx] == 2 && VectorQ[mnmx, NumericQ] :=
    Block[{bdsindex, im},
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];

        im = vdb["depthMap"[bdsindex, \[Gamma], mnmx[[1]], mnmx[[2]]]];

        im /; ImageQ[im]
    ]


iOpenVDBDepthImage[vdb_] := iOpenVDBDepthImage[vdb, Automatic]


iOpenVDBDepthImage[vdb_?carefulPixelGridQ, Automatic, args___] := iOpenVDBDepthImage[vdb, vdb["IndexBoundingBox"] -> $indexregime, args]


iOpenVDBDepthImage[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := iOpenVDBDepthImage[vdb, bspec -> $indexregime, opts]


iOpenVDBDepthImage[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ, args___] :=
    Block[{bds2d},
        bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];

        iOpenVDBDepthImage[vdb, Append[bds2d, int] -> regime, args] /; bounds2DQ[bds2d]
    ]


iOpenVDBDepthImage[vdb_, bdspec_] := iOpenVDBDepthImage[vdb, bdspec, 0.5]


iOpenVDBDepthImage[vdb_, bdspec_, \[Gamma]_] := iOpenVDBDepthImage[vdb, bdspec, \[Gamma], {0.2, 1.0}]


iOpenVDBDepthImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBDepthImage, 1];


SyntaxInformation[OpenVDBDepthImage] = {"ArgumentsPattern" -> {_, _., _., _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBDepthImage] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDepthImage[expr_, ___] /; messageGridQ[expr, OpenVDBDepthImage] = $Failed;


mOpenVDBDepthImage[expr_, ___] /; messagePixelGridQ[expr, OpenVDBDepthImage] = $Failed;


mOpenVDBDepthImage[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBDepthImage] = $Failed;


mOpenVDBDepthImage[_, _, \[Gamma]_, ___] /; !TrueQ[\[Gamma] > 0] :=
    (
        Message[OpenVDBDepthImage::gamma, \[Gamma], 3];
        $Failed
    )


mOpenVDBDepthImage[_, _, _, mnmx_] /; !MatchQ[mnmx, {a_, b_} /; 0 <= a <= b] :=
    (
        Message[OpenVDBDepthImage::range, mnmx, 4];
        $Failed
    )


mOpenVDBDepthImage[___] = $Failed;


OpenVDBDepthImage::gamma = "`1` at position `2` should be a positive number.";
OpenVDBDepthImage::range = "`1` at position `2` should be a list of two increasing non\[Hyphen]negative numbers.";


(* ::Section:: *)
(*Projection*)


(* ::Subsection::Closed:: *)
(*OpenVDBProjectionImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBProjectionImage[args___] /; !CheckArgs[OpenVDBProjectionImage[args], {1, 2}] = $Failed;


OpenVDBProjectionImage[args___] :=
    With[{res = iOpenVDBProjectionImage[args]},
        res /; res =!= $Failed
    ]


OpenVDBProjectionImage[args___] := mOpenVDBProjectionImage[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBProjectionImage*)


iOpenVDBProjectionImage[vdb_?carefulPixelGridQ, bds_?bounds3DQ -> regime_?regimeQ] :=
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


iOpenVDBProjectionImage[vdb_] := iOpenVDBProjectionImage[vdb, Automatic]


iOpenVDBProjectionImage[vdb_?carefulPixelGridQ, Automatic] := iOpenVDBProjectionImage[vdb, vdb["IndexBoundingBox"] -> $indexregime]


iOpenVDBProjectionImage[vdb_, bspec_List, opts:OptionsPattern[]] /; bounds3DQ[bspec] || intervalQ[bspec] := iOpenVDBProjectionImage[vdb, bspec -> $indexregime, opts]


iOpenVDBProjectionImage[vdb_?carefulPixelGridQ, int_?intervalQ -> regime_?regimeQ] :=
    Block[{bds2d},
        bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];

        iOpenVDBProjectionImage[vdb, Append[bds2d, int] -> regime] /; bounds2DQ[bds2d]
    ]


iOpenVDBProjectionImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBProjectionImage, 1];


SyntaxInformation[OpenVDBProjectionImage] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBProjectionImage] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBProjectionImage[expr_, ___] /; messageGridQ[expr, OpenVDBProjectionImage] = $Failed;


mOpenVDBProjectionImage[expr_, ___] /; messagePixelGridQ[expr, OpenVDBProjectionImage] = $Failed;


mOpenVDBProjectionImage[_, bbox_] /; message3DBBoxQ[bbox, OpenVDBProjectionImage] = $Failed;


mOpenVDBProjectionImage[___] = $Failed;


(* ::Section:: *)
(*Slice*)


(* ::Subsection::Closed:: *)
(*OpenVDBSliceImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBSliceImage] = {"MirrorSlice" -> False};


OpenVDBSliceImage[args___] /; !CheckArgs[OpenVDBSliceImage[args], {1, 3}] = $Failed;


OpenVDBSliceImage[args___] :=
    With[{res = iOpenVDBSliceImage[args]},
        res /; res =!= $Failed
    ]


OpenVDBSliceImage[args___] := mOpenVDBSliceImage[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBSliceImage*)


Options[iOpenVDBSliceImage] = Options[OpenVDBSliceImage];


iOpenVDBSliceImage[vdb_?carefulPixelGridQ, z_?NumericQ -> regime_?regimeQ, bds_?bounds2DQ, OptionsPattern[]] :=
    Block[{mirrorQ, threadedQ, zindex, bdsindex, im},
        mirrorQ = TrueQ[OptionValue["MirrorSlice"]];
        threadedQ = True;

        zindex = regimeConvert[vdb, z, regime -> $indexregime];
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];

        im = vdb["gridSliceImage"[zindex, bdsindex, mirrorQ, threadedQ]];

        im /; ImageQ[im]
    ]


iOpenVDBSliceImage[vdb_, z_?NumericQ, args___] := iOpenVDBSliceImage[vdb, z -> $indexregime, args]


iOpenVDBSliceImage[vdb_?carefulPixelGridQ, z_ -> regime_, Automatic, opts___] :=
    Block[{bbox},
        bbox = If[worldRegimeQ[regime],
            "WorldBoundingBox",
            "IndexBoundingBox"
        ];
        iOpenVDBSliceImage[vdb, z -> regime, Most[vdb[bbox]], opts]
    ]


iOpenVDBSliceImage[vdb_, z_, opts:OptionsPattern[]] := iOpenVDBSliceImage[vdb, z, Automatic, opts]


iOpenVDBSliceImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBSliceImage, 1];


SyntaxInformation[OpenVDBSliceImage] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSliceImage] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBSliceImage[expr_, ___] /; messageGridQ[expr, OpenVDBSliceImage] = $Failed;


mOpenVDBSliceImage[expr_, ___] /; messagePixelGridQ[expr, OpenVDBSliceImage] = $Failed;


mOpenVDBSliceImage[_, z_, ___] /; messageZSliceQ[z, OpenVDBSliceImage] = $Failed;


mOpenVDBSliceImage[_, _, bbox_, ___] /; message2DBBoxQ[bbox, OpenVDBSliceImage] = $Failed;


mOpenVDBSliceImage[___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBDynamicSliceImage*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDynamicSliceImage] = {DisplayFunction -> Identity, ImageSize -> Automatic};


OpenVDBDynamicSliceImage[args___] /; !CheckArgs[OpenVDBDynamicSliceImage[args], 1] = $Failed;


OpenVDBDynamicSliceImage[args___] :=
    With[{res = iOpenVDBDynamicSliceImage[args]},
        res /; res =!= $Failed
    ]


OpenVDBDynamicSliceImage[args___] := mOpenVDBDynamicSliceImage[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDynamicSliceImage*)


Options[iOpenVDBDynamicSliceImage] = Options[OpenVDBDynamicSliceImage];


iOpenVDBDynamicSliceImage[vdb_?carefulPixelGridQ, OptionsPattern[]] /; !emptyVDBQ[vdb] :=
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


iOpenVDBDynamicSliceImage[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBDynamicSliceImage] = {"ArgumentsPattern" -> {_, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDynamicSliceImage[expr_, ___] /; messageGridQ[expr, OpenVDBDynamicSliceImage] = $Failed;


mOpenVDBDynamicSliceImage[expr_, ___] /; messagePixelGridQ[expr, OpenVDBDynamicSliceImage] = $Failed;


mOpenVDBDynamicSliceImage[expr_, ___] /; messageNonEmptyGridQ[expr, OpenVDBDynamicSliceImage] = $Failed;


mOpenVDBDynamicSliceImage[___] = $Failed;
