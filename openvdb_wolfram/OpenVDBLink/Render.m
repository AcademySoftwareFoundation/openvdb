(* ::Package:: *)

(* ::Title:: *)
(*Render*)


(* ::Subtitle:: *)
(*Various rendering techniques on grids.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBLevelSetRender"]
PackageExport["OpenVDBLevelSetViewer"]


OpenVDBLevelSetRender::usage = "OpenVDBLevelSetRender[expr] returns a 3D render Image of an OpenVDB level set grid.";


OpenVDBLevelSetViewer::usage = "OpenVDBLevelSetViewer[expr] returns a manipulatable 3D render Image of an OpenVDB level set grid.";


(* ::Section:: *)
(*Shader and theme names*)


(* ::Subsection::Closed:: *)
(*Shaders*)


$dielectricList = {"Rubber", "Plastic", "Diffuse", "Glazed", "Satin", "Clay", "Matte", "Foil"};
$metalList = {"Gold", "Aluminum", "Brass", "Bronze", "Copper", "Electrum", "Iron", "Pewter"};
$rainbowList = {"Normal", "Position"};
$depthList = {"Depth"};


(* ::Subsection::Closed:: *)
(*Themes*)


$renderColorThemes = <|
                 (*     front face             back face           closed face              background *)
    "Default"    -> {RGBColor["#4D94FF"], RGBColor["#FFA24E"], RGBColor["#4D94FF"], RGBColor[1, 1, 1]},
    "Monochrome" -> {RGBColor[1, 1, 1],   RGBColor[1, 1, 1],   RGBColor[1, 1, 1],   RGBColor[0, 0, 0]},
    "Bold"       -> {ColorData[106, 2],   ColorData[106, 1],   ColorData[106, 2],   Lighter[ColorData[106, 4], 0.5]},
    "Cool"       -> {ColorData[107, 1],   ColorData[107, 4],   ColorData[107, 1],   RGBColor["#E6EBFF"]},
    "Neon"       -> {ColorData[109, 8],   ColorData[109, 2],   ColorData[109, 8],   ColorData[109, 3]},
    "Pastel"     -> {RGBColor["#D5B0F6"], RGBColor["#F5AE6F"], RGBColor["#D5B0F6"], RGBColor["#B9FFDB"]},
    "Soft"       -> {RGBColor["#CFCFE1"], RGBColor["#EBBBAC"], RGBColor["#CFCFE1"], RGBColor["#3C3C78"]},
    "Vibrant"    -> {ColorData[112, 2],   ColorData[112, 1],   ColorData[112, 2],   RGBColor["#FFDC80"]},
    "Warm"       -> {ColorData[113, 1],   ColorData[113, 2],   ColorData[113, 1],   RGBColor["#FFE7B7"]}
|>;


(* ::Section:: *)
(*OpenVDBLevelSetRender*)


(* ::Subsection::Closed:: *)
(*Main*)


Options[OpenVDBLevelSetRender] = Join[
    {Background -> Automatic, "ClosedClipping" -> False, "FrameTranslation" -> Automatic, ImageResolution -> Automatic,
        "IsoValue" -> 0.0, "OrthographicFrame" -> Automatic, PerformanceGoal :> $PerformanceGoal},
    Options[Graphics3D, {ImageSize, ViewAngle, ViewCenter, ViewPoint, ViewProjection, ViewRange, ViewVertical}]
];


OpenVDBLevelSetRender[args___] /; !CheckArgs[OpenVDBLevelSetRender[args], {1, 2}] = $Failed;


OpenVDBLevelSetRender[args___] :=
    With[{res = iLevelSetRender[args]},
        res /; res =!= $Failed
    ]


OpenVDBLevelSetRender[args___] := mLevelSetRender[args]


(* ::Subsection::Closed:: *)
(*iLevelSetRender*)


Options[iLevelSetRender] = Options[OpenVDBLevelSetRender];


iLevelSetRender[vdb_?OpenVDBScalarGridQ, shading_, opts:OptionsPattern[]] /; levelSetQ[vdb] :=
    Block[{ropts, res},
        ropts = parseRenderOptions[vdb, shading, opts];
        (
            res = oLevelSetRender[vdb, ropts];

            res /; res =!= $Failed

        ) /; AssociationQ[ropts]
    ]


iLevelSetRender[vdb_, opts:OptionsPattern[]] := iLevelSetRender[vdb, Automatic, opts]


iLevelSetRender[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iLevelSetRender, 1];


SyntaxInformation[OpenVDBLevelSetRender] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


addCodeCompletion[OpenVDBLevelSetRender][None, Join[Keys[$renderColorThemes], $dielectricList, $metalList], None];


(* ::Subsection::Closed:: *)
(*oLevelSetRender*)


Options[oLevelSetRender] = Options[iLevelSetRender];


oLevelSetRender[vdb_, ropts_] /; emptyVDBQ[vdb] || (!fogVolume[vdb] && vdb["BackgroundValue"] <= ropts["IsoValue"]) :=
    Block[{bg, res, ires},
        bg = RGBColor @@ ropts["Background"];
        res = ropts["Resolution"];
        ires = ropts["ImageResolution"];

        ConstantImage[bg, res, "Byte", ImageResolution -> ires]
    ]


oLevelSetRender[vdb_, ropts_] /; KeyExistsQ[materialParameters, ropts["Shader"]] :=
    Block[{ires, pbrparams, args, im},
        ires = ropts["ImageResolution"];
        args = Lookup[ropts, $PBRrenderLevelSetArgumentKeys];
        pbrparams = materialParameters[ropts["Shader"]];
        (
            im = vdb["renderGridPBR"[##]]& @@ Join[args, Values[pbrparams][[4 ;; -1]]];

            Image[im, ImageResolution -> ires] /; ImageQ[im]
        ) /; AssociationQ[pbrparams]
    ]


oLevelSetRender[vdb_, ropts_] :=
    Block[{ires, args, im},
        ires = ropts["ImageResolution"];
        args = Lookup[ropts, $renderLevelSetArgumentKeys];
        im = vdb["renderGrid"[##]]& @@ args;

        Image[im, ImageResolution -> ires] /; ImageQ[im]
    ]


oLevelSetRender[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Messages*)


mLevelSetRender[args___] := messageRenderFunction[OpenVDBLevelSetRender, args]


(* ::Section:: *)
(*OpenVDBLevelSetViewer*)


(* ::Subsection::Closed:: *)
(*Main*)


Options[OpenVDBLevelSetViewer] = Options[OpenVDBLevelSetRender];


OpenVDBLevelSetViewer[args___] /; !CheckArgs[OpenVDBLevelSetViewer[args], {1, 2}] = $Failed;


OpenVDBLevelSetViewer[args___] :=
    With[{res = iLevelSetViewer[args]},
        res /; res =!= $Failed
    ]


OpenVDBLevelSetViewer[args___] := mLevelSetViewer[args]


(* ::Subsection::Closed:: *)
(*iLevelSetViewer*)


Options[iLevelSetViewer] = Options[OpenVDBLevelSetViewer];


iLevelSetViewer[vdb_?OpenVDBScalarGridQ, shading_, opts:OptionsPattern[]] /; levelSetQ[vdb] :=
    Block[{ropts, args, im},
        ropts = parseRenderOptions[vdb, shading, opts];
        (
            iDynamicRender[vdb, shading, ropts, opts]

        ) /; AssociationQ[ropts]
    ]


iLevelSetViewer[vdb_, opts:OptionsPattern[]] := iLevelSetViewer[vdb, Automatic, opts]


iLevelSetViewer[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iLevelSetViewer, 1];


SyntaxInformation[OpenVDBLevelSetViewer] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


addCodeCompletion[OpenVDBLevelSetViewer][None, Join[Keys[$renderColorThemes], $dielectricList, $metalList], None];


(* ::Subsection::Closed:: *)
(*iDynamicRender*)


Options[iDynamicRender] = Options[OpenVDBLevelSetViewer];


iDynamicRender[vdb_, shading_, iropts_, opts:OptionsPattern[]] :=
    DynamicModule[{t, l, b, vp, ovp, vv, ovv, va, ova, vc, vr, sz, dx = 0.0, dy = 0.0, dz = 0.0, origva, origvc, origvr, origvp, origvv, mem, iso, oiso, mniso, mxiso, ir,
        td\[Delta], \[Delta], shader, oshader, vrng, ovrng, volQ, ivolQ, projection, oprojection, origprojection, oframe, ooframe, c1, oc1, c2, oc2, c3, oc3, bg, obg, clp, im, antialq, oantialq},
        {t, l, b} = Lookup[iropts, {"Translate", "LookAt", "Bounds"}];
        td\[Delta] = {0.005, 0.005, 0.1}*Max[Abs[Subtract @@@ b]];

        {vp, vv} = OptionValue[{ViewPoint, ViewVertical}];
        va = constructViewAngle[OptionValue[ViewAngle], t, l, b];
        vc = parseRenderViewCenter[OptionValue[ViewCenter], t, l, {{0, 1}, {0, 1}, {0, 1}}];

        origvc = vc;
        origvp = ovp = vp;
        origvv = ovv = vv;
        origva = ova = va;

        origvr = ovrng = initialViewRange[OptionValue[ViewRange], vp, vc, vv, va, b];
        volQ = ivolQ = TrueQ[OptionValue["ClosedClipping"]];

        ir = OptionValue[ImageResolution];
        sz = initialImageSize[iropts["ImageSize"], OptionValue[ImageSize]];
        oshader = canonicalizeShader[shading][[1]];

        projection = Replace[OptionValue[ViewProjection], Automatic -> "Perspective", {0}];
        origprojection = oprojection = projection;
        oframe = orthographicSphericalFrame[b];
        ooframe = oframe;

        {c1, c2, c3, bg} = If[TrueQ[customColorQ[oshader]],
            RGBColor @@@ Lookup[iropts, {"BaseColorFront", "BaseColorBack", "BaseColorClosed", "Background"}],
            $renderColorThemes["Default"]
        ];
        {oc1, oc2, oc3, obg} = {c1, c2, c3, bg};

        oiso = iropts["IsoValue"];
        {mniso, mxiso} = {-1, 1}Ramp[vdb["BackgroundValue"] - 1.5voxelSize[vdb]];
        mem = Quotient[vdb["MemoryUsage"], Replace[$ProcessorCount, Except[_Integer?Positive] -> 1, {0}]];

        oantialq = OptionValue[PerformanceGoal] =!= "Speed";

        Manipulate[
            Which[
                dz != 0,
                vp = zoomViewPoint[vp, vc, td\[Delta][[3]]dz, vv, va, b],
                dx != 0 || dy != 0,
                vc = translateViewCenter[vc, {td\[Delta][[1]]dx, td\[Delta][[2]]dy}]
            ];
            Overlay[
                {
                    Dynamic @ Image[im = iRender[
                        vdb, makeShader[shader, c1, c2, c3], "BoundingBox" -> b, ViewPoint -> vp, ViewVertical -> vv, ViewAngle -> va, ViewCenter -> vc,
                        ViewRange -> Scaled[vrng], ViewProjection -> projection, "OrthographicFrame" -> oframe, Background -> bg,
                        ImageSize -> dRenderImageSize[sz, \[Delta], OptionValue[PerformanceGoal], mem], "IsoValue" -> Clip[iso, .999{mniso, mxiso}],
                        PerformanceGoal -> If[antialq && !$ControlActiveSetting, "Quality", "Speed"], ImageResolution -> ir,
                        "ClosedClipping" -> volQ, opts
                    ], ImageSize -> sz],
                    Graphics3D[{},
                        Boxed -> False, Method -> {}, ImageSize -> Dynamic[sz],
                        ViewPoint -> Dynamic[vp], ViewVertical -> Dynamic[vv], ViewAngle -> Dynamic[va], ViewCenter -> Dynamic[vc]
                    ]
                },
                {1, 2},
                2
            ],
            OpenerView[{
            Style["Appearance", Medium],
                Column[{
                    Control[{{shader, oshader, "shading"},
                        KeyValueMap[#1 -> Row[{If[#2 === Automatic, Dynamic[c1], #2], ToLowerCase[#1]}, Spacer[2]]&, $dynamicRenderShaders],
                        ControlType -> PopupMenu
                    }],
                    Row[{
                        Control[{{antialq, oantialq, "anti aliasing"}, {True, False}}],
                        Control[{{volQ, ivolQ, "closed clipping"}, {True, False}}]
                    }, Spacer[2]],
                    Control[{{\[Delta], 1, "resolution"}, 0, 1}],
                    If[mxiso > 0, Control[{{iso, oiso, "iso\[Hyphen]value"}, mniso, mxiso}], Nothing],
                    Row[{
                        "themes",
                        Row[KeyValueMap[themeButton[ToLowerCase[#1], {c1, c2, c3, bg} = #2, Enabled -> Dynamic[customColorQ[shader]]]&, $renderColorThemes]]
                    }, Spacer[2]],
                    Dynamic @ Which[
                        !customColorQ[shader],
                        Row[{
                            Control[{{bg, obg, "background"}, obg, ControlType -> ColorSetter}]
                        }, Spacer[10]],
                        !volQ,
                        Row[{
                            Control[{{bg, obg, "background"}, obg, ControlType -> ColorSetter}],
                            Control[{{c1, oc1, "front face"}, oc1, ControlType -> ColorSetter}],
                            Control[{{c2, oc2, "back face"}, oc2, ControlType -> ColorSetter}], Spacer[3],
                            If[c1 =!= c2, themeButton["flip faces", If[c1 === c3, c3 = c2]; {c1, c2} = {c2, c1}], Nothing]
                        }, Spacer[1]],
                        c1 === c2 === c3,
                        Row[{
                            Control[{{bg, obg, "background"}, obg, ControlType -> ColorSetter}],
                            Control[{{c1, oc1, "front face"}, oc1, ControlType -> ColorSetter}],
                            Control[{{c2, oc2, "back face"}, oc2, ControlType -> ColorSetter}],
                            Control[{{c3, oc3, "closed face"}, oc3, ControlType -> ColorSetter}]
                        }, Spacer[1]],
                        True,
                        Grid[{{
                            Control[{{c1, oc1, "front face"}, oc1, ControlType -> ColorSetter}],
                            Control[{{c2, oc2, "back face"}, oc2, ControlType -> ColorSetter}], Spacer[3],
                            If[c1 =!= c2, Row[{themeButton["flip faces", If[c1 === c3, c3 = c2]; {c1, c2} = {c2, c1}], ""}], Nothing]
                        }, {
                            Control[{{bg, obg, "background"}, obg, ControlType -> ColorSetter}],
                            Control[{{c3, oc3, "closed face"}, oc3, ControlType -> ColorSetter}], Spacer[3],
                            If[c1 =!= c3, Row[{themeButton["set closed as front", c3 = c1], ""}], Nothing]
                        }},
                        Alignment -> {{Right, Right, Left, Left}, Automatic},
                        Spacings -> {1, Automatic}]
                    ]
                },
                Spacings -> {Automatic, {Automatic, Automatic, Automatic, 1, 1}}]
            }, Method -> "Active"],
            OpenerView[{
            Style["Orientation", Medium],
                Column[{
                    Control[{{vv, ovv, "view vertical"}, {{0,0,1} -> "(0,0,1)", {0,0,-1} -> "(0,0,\[Hyphen]1)", {1,0,0} -> "(1,0,0)", {-1,0,0} -> "(\[Hyphen]1,0,0)", {0,1,0} -> "(0,1,0)", {0,-1,0} -> "(0,\[Hyphen]1,0)"}, ControlType -> Setter}],
                    Control[{{vp, ovp, "view point"}, {{1.3,-2.4,2} -> "default", {-3,0,0} -> "left", {3,0,0} -> "right", {0,-3,0} -> "front", {0,3,0} -> "back", {0,0,-3} -> "below", {0,0,3} -> "above"}, ControlType -> Setter}],
                    Row[{
                        "view point\[ThinSpace]&\[ThinSpace]center",
                        Framed[
                            Grid[{{
                                    EventHandler[panSlider[Dynamic[dz, (dz = If[Abs[#] < .05, 0., #])&], Enabled -> Dynamic[projection =!= "Orthographic"]], {"MouseUp" :> (dz = 0)}, PassEventsDown -> True],
                                    EventHandler[panSlider[Dynamic[dx, (dx = If[Abs[#] < .05, 0., #])&]], {"MouseUp" :> (dx = 0)}, PassEventsDown -> True],
                                    EventHandler[panSlider[Dynamic[dy, (dy = If[Abs[#] < .05, 0., #])&]], {"MouseUp" :> (dy = 0)}, PassEventsDown -> True]
                                },
                                {"out\[ThinSpace]\[TwoWayRule]\[ThinSpace]in", "left\[ThinSpace]\[TwoWayRule]\[ThinSpace]right", "down\[ThinSpace]\[TwoWayRule]\[ThinSpace]up"}},
                                Spacings -> {1, 0.5}
                            ],
                            ContentPadding -> True,
                            FrameStyle -> None,
                            ImageMargins -> {{0,0},{2,0}}
                        ]
                    }, Spacer[2]]
                }]
            }, Method -> "Active"],
            OpenerView[{
            Style["Field of View", Medium],
                Column[{
                    Row[{
                        Control[{{vrng, ovrng, "clipping"}, 0, 1, ControlType -> IntervalSlider, MinIntervalSize -> 0.01, Method -> "Stop"}],
                        Control[{{volQ, ivolQ, "closed"}, {True, False}}]
                    }, Spacer[3]],
                    Dynamic @ If[projection =!= "Orthographic",
                        Control[{{va, ova, "FOV (rad)"}, 0, 75*\[Pi]/180}],
                        Control[{{oframe, ooframe, "frame width"}, 0, ooframe}]
                    ],
                    Control[{{projection, oprojection, "projection"}, {"Perspective" -> "perspective", "Orthographic" -> "orthographic"}}]
                }]
            }, Method -> "Active"],
            OpenerView[{
            Style["General", Medium],
                Row[{
                    Button["Copy image",
                        CopyToClipboard[im]],
                    Button["Copy view settings",
                        CopyToClipboard[copyString @ {If[projection =!= "Orthographic", ViewAngle -> va, "OrthographicFrame" -> oframe], ViewCenter -> vc,
                            ViewRange -> unscaledViewRange[vrng, vp, vc, vv, va, b], ViewPoint -> vp, ViewProjection -> projection, ViewVertical -> vv}]],
                    Button["Reset view settings",
                        va = origva; vc = origvc; vrng = origvr; vp = origvp; vv = origvv; projection = origprojection; oframe = ooframe;]
                }]
            }, Method -> "Active"]
        ],
        Initialization -> With[{args = Sequence @@ Append[vdb, $SessionID]}, iDynamicRenderTrack[args] = vdb],
        Deinitialization :> With[{args = Sequence @@ Append[vdb, $SessionID]}, If[Head[iDynamicRenderTrack[args]] =!= iDynamicRenderTrack, iDynamicRenderTrack[args]=.]]
    ]


(* ::Subsection::Closed:: *)
(*iRender*)


Options[iRender] = Options[iLevelSetRender];


iRender[args__] :=
    With[{res = Quiet[iLevelSetRender[args]]},
        res /; ImageQ[res]
    ]


iRender[___] = Image[{{1}}];


(* ::Subsection::Closed:: *)
(*Utilities*)


Scan[(customColorQ[#] = True )&, $dielectricList];
Scan[(customColorQ[#] = False)&, $metalList];
Scan[(customColorQ[#] = False)&, $rainbowList];
Scan[(customColorQ[#] = True)&, $depthList];


$dynamicRenderShaders := $dynamicRenderShaders = Association @ Join[
    # -> Automatic& /@ $dielectricList,
    # -> materialParameters[#]["BaseColorFront"]& /@ $metalList,
    # -> $rainbowSwatch& /@ $rainbowList,
    # -> $depthSwatch& /@ $depthList
]


dynamicRenderShaders = $dynamicRenderShaders[If[ListQ[#], First[#, ""], #]]&;


$rainbowSwatch = Image[ImagePad[Image[Table[With[{r = Sqrt[x^2 + y^2], \[Theta] = ArcTan[.0000001-y, x]}, Hue[\[Theta]/(2\[Pi])+1/2., Clip[r,{0,1}],1]], {x, -1., 1., .1}, {y, -1., 1., .1}]], 1], "Byte", ImageSize -> 11];


$depthSwatch = Image[ImagePad[Image[Table[With[{r = Sqrt[x^2 + y^2], \[Theta] = ArcTan[.0000001-y, x]}, Hue[0, 0, Clip[1-r, {0,1}]^0.5]], {x, -1., 1., .1}, {y, -1., 1., .1}]], 1], "Byte", ImageSize -> 11];


makeShader[shader_, colors__] :=
    If[TrueQ[customColorQ[shader]] || !KeyExistsQ[materialParameters, shader],
        {shader, colors},
        Prepend[Lookup[materialParameters[shader], {"BaseColorFront", "BaseColorBack", "BaseColorClosed"}], shader]
    ]


initialViewRange[{min_?NumericQ, max_?NumericQ}, vp_, vc_, vv_, va_, bds_] /; min <= max :=
    Block[{t, l, vr},
        t = parseRenderViewPoint[vp, bds];
        l = parseRenderViewCenter[vc, t, vv, va, bds];
        vr = parseRenderViewRange[Scaled[{0, 1}], t, l, bds];
        Clip[Rescale[{min, max}, vr], {0, 1}]
    ]
initialViewRange[___] = {0, 1};


unscaledViewRange[vrng_, vp_, vc_, vv_, va_, bds_] :=
    Block[{t, l},
        t = parseRenderViewPoint[vp, bds];
        l = parseRenderViewCenter[vc, t, vv, va, bds];
        parseRenderViewRange[Scaled[vrng], t, l, bds]
    ]


copyString[expr_List] := StringTake[ToString[expr, InputForm], {2, -2}]


copyString[expr_] := ToString[expr, InputForm]


initialImageSize[{sx_, _}, Automatic] := {sx, sx}
initialImageSize[sz_, _] := sz


dRenderImageSize[sz_, \[Delta]_, pgoal_, mem_] :=
    With[{s = sz*\[Delta]},
        Clip[Round[dPGoalFactor[pgoal, s, mem]*s], {1, \[Infinity]}]
    ]


(* ::Text:: *)
(*TODO figure out when rotating becomes laggy. I think it's some function of image size, vdb memory footprint, and average depth of the first layer.*)


dPGoalFactor["Speed", sz_, mem_] :=
    If[TrueQ[(Times @@ sz) > Min[450^2, Divide[500000000.*720^2, mem]] && $ControlActiveSetting],
        0.5,
        1.0
    ]
dPGoalFactor[__] = 1.0;


panSlider[var_, opts___] := Slider[var, {-1, 1}, opts, Appearance -> "UpArrow", ImageSize -> {75, 22}]


orthographicSphericalFrame[bds_] :=
    With[{reg = BoundingRegion[Tuples[bds], "MinBall"]},
        2reg[[2]] /; MatchQ[reg, Ball[_, _Real]]
    ]


orthographicSphericalFrame[___] = $Failed;


renderColor[__, "Normal"|"NormalClosed"|"Position"|"PositionClosed"] = White;
renderColor[c1_, c1_, _] := c1
renderColor[c1_, c2_, _] := FaceForm[c1, c2]


SetAttributes[themeButton, HoldRest];
themeButton[label_, action_, opts___] := Button[Style[label, Small], action, opts, Appearance -> "Palette"]


translateViewCenter[{vc_, {x_, y_}}, {dx_, dy_}] := {vc, {x+dx, y+dy}}
translateViewCenter[vc_, {dx_, dy_}] := {vc, {0.5+dx, 0.5+dy}}


zoomViewPoint[vp_, vc_, dz_, vv_, va_, bds_] :=
    Block[{translate, lookat, dir, center, mx},
        translate = parseRenderViewPoint[vp, bds];
        lookat = parseRenderViewCenter[vc, translate, vv, va, bds];
        dir = Min[.1dz*Power[Norm[Subtract[lookat, translate]], 1.25], 1.5]Normalize[Subtract[lookat, translate]];

        translate += dir;
        lookat    += dir;

        center = Mean /@ bds;
        mx = Replace[Max[Abs[Subtract @@@ bds]], _?NonPositive -> 1.0, {0}];

        Divide[Subtract[translate, center], mx]
    ]


(* ::Subsection::Closed:: *)
(*Messages*)


mLevelSetViewer[args___] := messageRenderFunction[OpenVDBLevelSetViewer, args]


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*$renderLevelSetArgumentKeys*)


$renderLevelSetArgumentKeys = {"IsoValue", "BaseColorFront", "BaseColorBack", "BaseColorClosed", "Background", "Translate", "LookAt", "Up",
    "Range", "FOV", "Shader", "Camera", "Samples", "Resolution", "Frame", "DepthParameters", "Lighting", "Step", "IsClosed"};


$PBRrenderLevelSetArgumentKeys = {"IsoValue", "Background", "Translate", "LookAt", "Up",
    "Range", "FOV", "Camera", "Samples", "Resolution", "Frame", "IsClosed", "BaseColorFront", "BaseColorBack", "BaseColorClosed"};


(* ::Subsection::Closed:: *)
(*Option parsing utilities*)


(* ::Subsubsection::Closed:: *)
(*parseRenderOptions*)


Options[parseRenderOptions] = Join[Options[iLevelSetRender], {"BoundingBox" -> Automatic}];


parseRenderOptions[vdb_, shading_, opts:OptionsPattern[]] :=
    Block[{res},
        res = iparseRenderOptions[vdb, shading, opts];

        res /; AssociationQ[res] && NoneTrue[res, FailureQ]
    ]


parseRenderOptions[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*iparseRenderOptions*)


Options[iparseRenderOptions] = Options[parseRenderOptions];


iparseRenderOptions[vdb_, shading_, OptionsPattern[]] :=
    Block[{bds, translate, lookat1, \[Alpha], lookat, up, transdist, depthdata, imgresolution, shader, ropts},
        bds = parseBoundingBox[vdb, OptionValue["BoundingBox"]];

        translate = parseRenderViewPoint[OptionValue[ViewPoint], bds];
        up = parseRenderViewVertical[OptionValue[ViewVertical], OptionValue[ViewPoint]];

        lookat1 = parseRenderViewCenterNoOffset[OptionValue[ViewCenter], bds];
        \[Alpha] = constructViewAngle[OptionValue[ViewAngle], translate, lookat1, bds];
        lookat = parseRenderViewCenter[OptionValue[ViewCenter], translate, up, \[Alpha], bds];

        shader = canonicalizeShader[shading];

        depthdata = parseDepthParameters[shader];
        imgresolution = parseRenderImageResolution[OptionValue[ImageResolution]];

        <|
            "Bounds" -> bds,
            "IsoValue" -> parseIsoValue[OptionValue["IsoValue"]],
            "Background" -> parseRenderBackgroundColor[OptionValue[Background], shader],
            "Translate" -> translate,
            "LookAt" -> lookat,
            "Up" -> up,
            "Range" -> parseRenderViewRange[OptionValue[ViewRange], translate, lookat, bds],
            "FOV" -> parseRenderViewAngle[OptionValue[ViewAngle], translate, lookat, bds],
            "Shader" -> parseRenderShader[First[shader, $Failed]],
            "Camera" -> parseRenderViewProjection[OptionValue[ViewProjection], OptionValue[ViewPoint]],
            "Samples" -> parseRenderPerformance[OptionValue[PerformanceGoal]],
            "Resolution" -> parseRenderImageSize[OptionValue[ImageSize], imgresolution, translate, up, bds],
            "ImageSize" -> parseRenderUnscaledImageSize[OptionValue[ImageSize], translate, up, bds],
            "ImageResolution" -> imgresolution,
            "Frame" -> parseOrthographicFrame[OptionValue[ViewProjection], OptionValue["OrthographicFrame"], translate, lookat, up, bds],
            "DepthParameters" -> depthdata,
            "Lighting" -> RotationTransform[0.25, {-1, 1, 1}][translate - lookat],
            "Step" -> {1.0, 2.0},
            "BaseColorFront" -> parseRenderColor[shader],
            "BaseColorBack" -> parseRenderColor2[shader],
            "BaseColorClosed" -> parseRenderColor3[shader],
            "IsClosed" -> TrueQ[OptionValue["ClosedClipping"]]
        |>
    ]


(* ::Subsubsection::Closed:: *)
(*renderFailureOption*)


renderFailureOption[assoc_] :=
    (
        If[FailureQ[assoc["Bounds"]], Return["BoundingBox"]];
        If[FailureQ[assoc["IsoValue"]], Return["IsoValue"]];

        If[FailureQ[assoc["Shader"]], Return["Shader"]];
        If[FailureQ[assoc["Background"]], Return[Background]];

        If[FailureQ[assoc["Translate"]], Return[ViewPoint]];
        If[FailureQ[assoc["Up"]], Return[ViewVertical]];
        If[FailureQ[assoc["FOV"]], Return[ViewAngle]];
        If[FailureQ[assoc["LookAt"]], Return[ViewCenter]];
        If[FailureQ[assoc["Range"]], Return[ViewRange]];

        If[FailureQ[assoc["Camera"]], Return[ViewProjection]];
        If[FailureQ[assoc["Samples"]], Return[PerformanceGoal]];
        If[FailureQ[assoc["ImageResolution"]], Return[ImageResolution]];
        If[FailureQ[assoc["ImageSize"]], Return[ImageSize]];

        $Failed
    )


(* ::Subsubsection::Closed:: *)
(*canonicalizeShader*)


$defaultShaderSpec = "Rubber";
$defaultColorSpec = "Default";


canonicalizeShader[shader_String] /; KeyExistsQ[materialParameters, shader] := Prepend[Lookup[materialParameters[shader], {"BaseColorFront", "BaseColorBack", "BaseColorClosed"}], shader]


canonicalizeShader[theme_String] /; KeyExistsQ[$renderColorThemes, theme] := {$defaultShaderSpec, theme}


canonicalizeShader[shader:("Depth"|{"Depth", _?NumericQ, __})] := {shader, RGBColor[1, 1, 1], RGBColor[1, 1, 1], RGBColor[1, 1, 1]}


canonicalizeShader[{shader_, theme_String}] := {shader, theme}


canonicalizeShader[{shader_, icolor_?ColorQ}] :=
    With[{color = ColorConvert[icolor, "RGB"][[1 ;; 3]]},
        {shader, color, color, color}
    ]


canonicalizeShader[{shader_, icolorf_?ColorQ, icolorb_?ColorQ}] :=
    With[{
            colorf = ColorConvert[icolorf, "RGB"][[1 ;; 3]],
            colorb = ColorConvert[icolorb, "RGB"][[1 ;; 3]]
        },
        {shader, colorf, colorb, colorf}
    ]


canonicalizeShader[{shader_, colorf_?ColorQ, colorb_?ColorQ, colorc_?ColorQ}] :=
    {shader, ColorConvert[colorf, "RGB"][[1 ;; 3]], ColorConvert[colorb, "RGB"][[1 ;; 3]], ColorConvert[colorc, "RGB"][[1 ;; 3]]}


canonicalizeShader[Automatic] = {$defaultShaderSpec, $defaultColorSpec};


canonicalizeShader[{colors__?ColorQ}] := canonicalizeShader[{$defaultShaderSpec, colors}]


canonicalizeShader[color_?ColorQ] := canonicalizeShader[{$defaultShaderSpec, color}]


canonicalizeShader[shader_] := {shader, $defaultColorSpec}


canonicalizeShader[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseBoundingBox*)


parseBoundingBox[vdb_, bds_] := If[bds === Automatic, vdb["WorldBoundingBox"], bds]


(* ::Subsubsection::Closed:: *)
(*parseIsoValue*)


parseIsoValue[Automatic] = 0.0;
parseIsoValue[x_?realQ] := x
parseIsoValue[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderColor*)


parseRenderColor[{"Normal"|"NormalClosed"|"Position"|"PositionClosed", __}] = {1.0, 1.0, 1.0};
parseRenderColor[{_, theme_String}] :=
    With[{color = $renderColorThemes[theme][[1]]},
        List @@ color /; ColorQ[color]
    ]
parseRenderColor[{_, color_?ColorQ, _, _}] := List @@ color
parseRenderColor[___] = $Failed;


parseRenderColor2[{"Normal"|"NormalClosed"|"Position"|"PositionClosed", __}] = {1.0, 1.0, 1.0};
parseRenderColor2[{_, theme_String}] :=
    With[{color = $renderColorThemes[theme][[2]]},
        List @@ color /; ColorQ[color]
    ]
parseRenderColor2[{_, _, color_?ColorQ, _}] := List @@ color
parseRenderColor2[___] = $Failed;


parseRenderColor3[{"Normal"|"NormalClosed"|"Position"|"PositionClosed", __}] = {1.0, 1.0, 1.0};
parseRenderColor3[{_, theme_String}] :=
    With[{color = $renderColorThemes[theme][[3]]},
        List @@ color /; ColorQ[color]
    ]
parseRenderColor3[{_, _, _, color_?ColorQ}] := List @@ color
parseRenderColor3[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderBackgroundColor*)


parseRenderBackgroundColor[None, shader_] := parseRenderBackgroundColor[White, shader]
parseRenderBackgroundColor[Automatic, shader:{"Depth"|{"Depth", ___}|"DepthClosed"|{"DepthClosed", ___}, __}] := parseRenderBackgroundColor[Black, shader]
parseRenderBackgroundColor[Automatic, shader_] := parseRenderBackgroundColor[White, shader]
parseRenderBackgroundColor[Automatic, {shader_, t_String}] /; KeyExistsQ[$renderColorThemes, t] := parseRenderBackgroundColor[$renderColorThemes[t][[4]], {shader, t}]
parseRenderBackgroundColor[color_?ColorQ, __] := List @@ ColorConvert[color, "RGB"][[1 ;; 3]]
parseRenderBackgroundColor[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewPoint*)


canonicalizeViewPoint[Left]  = {-2,0,0};
canonicalizeViewPoint[Right] = { 2,0,0};
canonicalizeViewPoint[Front] = {0,-2,0};
canonicalizeViewPoint[Back]  = {0, 2,0};
canonicalizeViewPoint[Below] = {0,0,-2};
canonicalizeViewPoint[Above] = {0,0, 2};
canonicalizeViewPoint[vp_] := vp


parseRenderViewPoint[vp_List, bds_?bounds3DQ] /; VectorQ[vp, NumericQ] && Length[vp] === 3 :=
    Block[{vpfinite, mx},
        vpfinite = vp /. inf_DirectedInfinity :> Sign[inf]*1000;
        mx = Replace[Max[Abs[Subtract @@@ bds]], _?NonPositive -> 1.0, {0}];

        mx * vpfinite + (Mean /@ bds)
    ]


parseRenderViewPoint[dir:(Left|Right|Front|Back|Below|Above),  bds_] := parseRenderViewPoint[canonicalizeViewPoint[dir], bds]


parseRenderViewPoint[dirs:{(Left|Right|Front|Back|Below|Above)..}, bds_] :=
    Block[{canons, vps},
        canons = canonicalizeViewPoint /@ dirs;
        (
            vps = GatherBy[canons, Unitize][[All, -1]];

            Total[parseRenderViewPoint[canonicalizeViewPoint[#], bds]& /@ vps]

        ) /; MatrixQ[canons]
    ]


parseRenderViewPoint[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewCenter*)


parseRenderViewCenter[{vc_List, {ox_?NumericQ, oy_?NumericQ}}, translate_, up_, \[Alpha]_, bds_?bounds3DQ] :=
    Block[{lookat = parseRenderViewCenterNoOffset[vc, bds], v, cross, rot},
        (
            v = translate-lookat;
            cross = Cross[v, up];

            rot = RotationTransform[(oy-0.5)*\[Alpha], cross, translate] @* RotationTransform[(ox-0.5)*\[Alpha], up, translate];

            rot[lookat]

        ) /; lookat =!= $Failed
    ]
parseRenderViewCenter[vc_, __, bds_] := parseRenderViewCenterNoOffset[vc, bds];


parseRenderViewCenterNoOffset[Automatic, bds_] := parseRenderViewCenterNoOffset[{0.5, 0.5, 0.5}, bds]
parseRenderViewCenterNoOffset[vc_List, bds_?bounds3DQ] /; VectorQ[vc, NumericQ] && Length[vc] === 3 := MapThread[Dot, {Transpose[{1-vc, vc}], bds}]
parseRenderViewCenterNoOffset[{vc_List, _}, bds_?bounds3DQ] := parseRenderViewCenterNoOffset[vc, bds]
parseRenderViewCenterNoOffset[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewVertical*)


parseRenderViewVertical[vv_List, vp_] /; VectorQ[vp, NumericQ] && Length[vp] === 3 :=
    If[degenerateViewVerticalQ[vv, vp],
        fixDegenerateViewVertical[vp],
        vv
    ]
parseRenderViewVertical[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewRange*)


parseRenderViewRange[Automatic|All, vp_, vc_, bds_] :=
    Block[{corners, inview, far},
        corners = Tuples[bds];
        inview = Pick[corners, RegionMember[HalfSpace[vp-vc, vp], corners]];
        If[Length[inview] == 0,
            parseRenderViewRange[{0.0, 0.0}, vp, vc, bds],
            far = Sqrt[Max[Total[(Transpose[inview] - vp)^2]]];
            parseRenderViewRange[{0.0, far}, vp, vc, bds]
        ]
    ]


parseRenderViewRange[{min_?NumericQ, max_?NumericQ}, ___] /; min <= max :=
    Block[{clip},
        clip = Clip[{min, max}, {.001, \[Infinity]}];
        If[Equal @@ clip,
            clip + {0, 0.001},
            clip
        ]
    ]


parseRenderViewRange[Scaled[vrng:{min_?NumericQ, max_?NumericQ}], vp_, vc_, bds_] /; min <= max :=
    Block[{padding, vrmin, vrmax},
        padding = 0.02*Max[Abs[Subtract @@@ bds]];
        vrmin = Clip[SignedRegionDistance[Cuboid @@ Transpose[bds], vp] - padding, {0.1, \[Infinity]}];
        vrmax = Ramp[Sqrt[Max[Total[Subtract[Transpose[Tuples[bds]], vp]^2, {1}]]] + .11];

        vrng*(vrmax-vrmin) + vrmin
    ]


parseRenderViewRange[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewAngle*)


$renderFocalLength = 50.0;


parseRenderViewAngle[args__] :=
    With[{aperture = viewAngleAperture[args]},
        {aperture, $renderFocalLength} /; aperture =!= $Failed && NumericQ[$renderFocalLength]
    ]
parseRenderViewAngle[___] = $Failed;


viewAngleAperture[args__] :=
    With[{\[Alpha] = constructViewAngle[args]},
        2$renderFocalLength*Tan[Clip[\[Alpha], {.001, 3.14059}]/2] /; \[Alpha] =!= $Failed
    ];
viewAngleAperture[___] = $Failed;


constructViewAngle[All, translate_, lookat_, bds_] :=
    With[{v = lookat - translate},
        2.0*Max[VectorAngle[v, # - translate]& /@ Tuples[bds]]
    ];
constructViewAngle[Automatic, args__] :=
    With[{\[Alpha] = constructViewAngle[All, args]},
        Min[35.0*Degree, \[Alpha]] /; \[Alpha] =!= $Failed
    ];
constructViewAngle[\[Alpha]_?NumericQ, args__] /; 0 <= \[Alpha] <= \[Pi] := \[Alpha]
constructViewAngle[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderShader*)


parseRenderShader["Diffuse"] = 0;


parseRenderShader["Matte"] = 1;


parseRenderShader["Normal"] = 2;


parseRenderShader["Position"] = 3;


parseRenderShader["Depth"] = 4;
parseRenderShader[{"Depth", ___}] = parseRenderShader["Depth"];


parseRenderShader[shader_String] /; KeyExistsQ[materialParameters, shader] := shader


parseRenderShader[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderPerformance*)


parseRenderPerformance["Speed"] = 1;
parseRenderPerformance["Quality"] = 9;
parseRenderPerformance[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseDepthParameters*)


$imin = 0.0;
$imax = 1.0;
$gamma = 1.0;


parseDepthParameters[{{"Depth"|"DepthClosed", args___}, __}] := iParseDepthParameters[args]
parseDepthParameters[___] = {$imin, $imax, $gamma};


iParseDepthParameters[] = {$imin, $imax, $gamma};
iParseDepthParameters[gamma_?Positive] := {$imin, $imax, gamma}
iParseDepthParameters[gamma_?Positive, {imin_, imax_}] /; 0 <= imin <= imax <= 1 := {imin, imax, gamma}
iParseDepthParameters[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderViewProjection*)


parseRenderViewProjection["Perspective", _]  = 0;
parseRenderViewProjection["Orthographic", _] = 1;
parseRenderViewProjection[Automatic, vp_] :=
    With[{p = If[FreeQ[vp, _DirectedInfinity], "Perspective", "Orthographic"]},
        parseRenderViewProjection[p, vp]
    ]
parseRenderViewProjection[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*parseRenderImageResolution*)


parseRenderImageResolution[Automatic] := imageResolution[]
parseRenderImageResolution[x_?Positive] := x
parseRenderImageResolution[___] = $Failed;


imageResolution[] := Replace[Max[Quiet[CurrentValue[{"ConnectedDisplays", "Resolution"}]]], Except[_Real|_Integer] -> 72., {0}]


(* ::Subsubsection::Closed:: *)
(*parseRenderImageSize*)


parseRenderImageSize[x:(_Integer|Automatic), args__] := parseRenderImageSize[{x, Automatic}, args]
parseRenderImageSize[{x_Integer?Positive, y_Integer?Positive}, res_, __] := res/72*{x, y}
parseRenderImageSize[{x_, y_}, res_, vp_List, vv_List, bds_?bounds3DQ] :=
    Block[{aspectratio, w, h},
        aspectratio = 1.0;(*viewPointAspectRatio[bds, vp, vv];*)
        w = Round[Replace[x, Automatic -> 360*res/72, {0}]];
        h = Round[Replace[y, Automatic -> w*aspectratio, {0}]];

        {w, h}
    ]
parseRenderImageSize[___] = $Failed


viewPointAspectRatio[bds_, vp_, vv_] :=
    Block[{rot, proj1, up2d, proj2},
        rot = RotationTransform[{vp-(Mean /@ bds), {0,0,1}}];
        proj1 = rot[Tuples[bds]][[All, 1;;2]];
        up2d = rot[vv][[1 ;; 2]];

        proj2 = RotationTransform[{up2d, {0, 1}}][proj1];
        Divide @@ Clip[Reverse[Abs[Subtract @@@ CoordinateBounds[proj2]]], {0.001, \[Infinity]}]
    ]


(* ::Subsubsection::Closed:: *)
(*parseRenderUnscaledImageSize*)


parseRenderUnscaledImageSize[x:(_Integer|Automatic), args__] := parseRenderUnscaledImageSize[{x, Automatic}, args]
parseRenderUnscaledImageSize[{x_Integer?Positive, y_Integer?Positive}, __] := {x, y}
parseRenderUnscaledImageSize[{x_, y_}, vp_List, vv_List, bds_?bounds3DQ] :=
    Block[{aspectratio, w, h},
        aspectratio = viewPointAspectRatio[bds, vp, vv];
        w = Round[Replace[x, Automatic -> 360, {0}]];
        h = Round[Replace[y, Automatic -> w*aspectratio, {0}]];

        {w, h}
    ]
parseRenderUnscaledImageSize[___] = $Failed


viewPointAspectRatio[___] = 1.0;


(* ::Subsubsection::Closed:: *)
(*parseOrthographicFrame*)


parseOrthographicFrame["Orthographic", x_?NumericQ, __] := Clip[x, {0.001, \[Infinity]}]
parseOrthographicFrame["Orthographic", Automatic, vp_, vc_, vv_, bds_] :=
    Block[{rot, proj1, up2d, proj2},
        rot = RotationTransform[{vp-vc, {0,0,1}}];
        proj1 = rot[Tuples[bds]][[All, 1;;2]];
        up2d = rot[vv][[1 ;; 2]];

        proj2 = RotationTransform[{up2d, {0, 1}}][proj1];
        Clip[Abs[Subtract @@@ CoordinateBounds[proj2]][[1]], {0.001, \[Infinity]}]
    ]
parseOrthographicFrame["Perspective"|Automatic, __] = 1.0;
parseOrthographicFrame[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Degenerate ViewVertical*)


degenerateViewVerticalQ[vv_, vp_] := Chop[Norm[Cross[vv, canonicalizeViewPoint[vp]]]] === 0


fixDegenerateViewVertical[vp_] := Normalize[RotateLeft[canonicalizeViewPoint[vp]]]


(* ::Subsection::Closed:: *)
(*MaterialShading parameters*)


(* ::Text:: *)
(*Found through ctrl-shift-e on MaterialShading[] outputs.*)


materialParameters = <||>;


(* ::Subsubsection::Closed:: *)
(*Aluminum*)


materialParameters["Aluminum"] = <|
    "BaseColorFront" -> RGBColor[0.95, 0.95, 0.95],
    "BaseColorBack" -> RGBColor[0.95, 0.95, 0.95],
    "BaseColorClosed" -> RGBColor[0.95, 0.95, 0.95],
    "MetallicCoefficient" -> 0.8,
    "RoughnessCoefficient" -> 0.75,
    "SpecularAnisotropyCoefficient" -> 0.6,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Brass*)


materialParameters["Brass"] = <|
    "BaseColorFront" -> RGBColor[0.9, 0.855, 0.45],
    "BaseColorBack" -> RGBColor[0.9, 0.855, 0.45],
    "BaseColorClosed" -> RGBColor[0.9, 0.855, 0.45],
    "MetallicCoefficient" -> 0.8,
    "RoughnessCoefficient" -> 0.65,
    "SpecularAnisotropyCoefficient" -> 0.4,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Bronze*)


materialParameters["Bronze"] = <|
    "BaseColorFront" -> RGBColor[0.9, 0.68625, 0.45],
    "BaseColorBack" -> RGBColor[0.9, 0.68625, 0.45],
    "BaseColorClosed" -> RGBColor[0.9, 0.68625, 0.45],
    "MetallicCoefficient" -> 0.8,
    "RoughnessCoefficient" -> 0.65,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Copper*)


materialParameters["Copper"] = <|
    "BaseColorFront" -> RGBColor[1.0, 0.65, 0.5],
    "BaseColorBack" -> RGBColor[1.0, 0.65, 0.5],
    "BaseColorClosed" -> RGBColor[1.0, 0.65, 0.5],
    "MetallicCoefficient" -> 0.8,
    "RoughnessCoefficient" -> 0.65,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Electrum*)


materialParameters["Electrum"] = <|
    "BaseColorFront" -> RGBColor[0.9, 0.774, 0.45],
    "BaseColorBack" -> RGBColor[0.9, 0.774, 0.45],
    "BaseColorClosed" -> RGBColor[0.9, 0.774, 0.45],
    "MetallicCoefficient" -> 0.7,
    "RoughnessCoefficient" -> 0.7,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Gold*)


materialParameters["Gold"] = <|
    "BaseColorFront" -> RGBColor[1.0, 0.75, 0.0],
    "BaseColorBack" -> RGBColor[1.0, 0.75, 0.0],
    "BaseColorClosed" -> RGBColor[1.0, 0.75, 0.0],
    "MetallicCoefficient" -> 0.8,
    "RoughnessCoefficient" -> 0.65,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Iron*)


materialParameters["Iron"] = <|
    "BaseColorFront" -> RGBColor[0.6, 0.576, 0.54],
    "BaseColorBack" -> RGBColor[0.6, 0.576, 0.54],
    "BaseColorClosed" -> RGBColor[0.6, 0.576, 0.54],
    "MetallicCoefficient" -> 0.7,
    "RoughnessCoefficient" -> 0.6,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Pewter*)


materialParameters["Pewter"] = <|
    "BaseColorFront" -> RGBColor[0.9, 0.864, 0.81],
    "BaseColorBack" -> RGBColor[0.9, 0.864, 0.81],
    "BaseColorClosed" -> RGBColor[0.9, 0.864, 0.81],
    "MetallicCoefficient" -> 1,
    "RoughnessCoefficient" -> 0.75,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Silver*)


materialParameters["Silver"] = <|
    "BaseColorFront" -> RGBColor[1.0, 1.0, 1.0],
    "BaseColorBack" -> RGBColor[1.0, 1.0, 1.0],
    "BaseColorClosed" -> RGBColor[1.0, 1.0, 1.0],
    "MetallicCoefficient" -> 1,
    "RoughnessCoefficient" -> 0.75,
    "SpecularAnisotropyCoefficient" -> 0.3,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Clay*)


materialParameters["Clay"] = <|
    "BaseColorFront" -> RGBColor[0.8, 0.352, 0.16],
    "BaseColorBack" -> RGBColor[0.8, 0.352, 0.16],
    "BaseColorClosed" -> RGBColor[0.8, 0.352, 0.16],
    "MetallicCoefficient" -> 0,
    "RoughnessCoefficient" -> 0,
    "SpecularAnisotropyCoefficient" -> 0.0,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 0.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Foil*)


materialParameters["Foil"] = <|
    "BaseColorFront" -> RGBColor[0.5, 1.0, 0.0],
    "BaseColorBack" -> RGBColor[0.5, 1.0, 0.0],
    "BaseColorClosed" -> RGBColor[0.5, 1.0, 0.0],
    "MetallicCoefficient" -> 0.5,
    "RoughnessCoefficient" -> 0.6,
    "SpecularAnisotropyCoefficient" -> 0.0,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.6,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.75
|>;


(* ::Subsubsection::Closed:: *)
(*Glazed*)


materialParameters["Glazed"] = <|
    "BaseColorFront" -> RGBColor[1.0, 0.24, 0.0],
    "BaseColorBack" -> RGBColor[1.0, 0.24, 0.0],
    "BaseColorClosed" -> RGBColor[1.0, 0.24, 0.0],
    "MetallicCoefficient" -> 0.5,
    "RoughnessCoefficient" -> 0.6,
    "SpecularAnisotropyCoefficient" -> 0.6,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.2,
    "CoatAnisotropyCoefficient" -> 0.6,
    "CoatReflectance" -> 0.6,
    "SpecularColorMultiplier" -> 1.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.75
|>;


(* ::Subsubsection::Closed:: *)
(*Plastic*)


materialParameters["Plastic"] = <|
    "BaseColorFront" -> RGBColor[0.3, 0.58, 1.0],
    "BaseColorBack" -> RGBColor[0.3, 0.58, 1.0],
    "BaseColorClosed" -> RGBColor[0.3, 0.58, 1.0],
    "MetallicCoefficient" -> 0.0,
    "RoughnessCoefficient" -> 1.0,
    "SpecularAnisotropyCoefficient" -> 0.0,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.3,
    "CoatAnisotropyCoefficient" -> 0.5,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.75
|>;


(* ::Subsubsection::Closed:: *)
(*Rubber*)


materialParameters["Rubber"] = <|
    "BaseColorFront" -> RGBColor[0.5, 0.5, 0.5],
    "BaseColorBack" -> RGBColor[0.5, 0.5, 0.5],
    "BaseColorClosed" -> RGBColor[0.5, 0.5, 0.5],
    "MetallicCoefficient" -> 0.2,
    "RoughnessCoefficient" -> 0.6,
    "SpecularAnisotropyCoefficient" -> 0.0,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Satin*)


materialParameters["Satin"] = <|
    "BaseColorFront" -> RGBColor[0.75, 0.5, 1.0],
    "BaseColorBack" -> RGBColor[0.75, 0.5, 1.0],
    "BaseColorClosed" -> RGBColor[0.75, 0.5, 1.0],
    "MetallicCoefficient" -> 0.5,
    "RoughnessCoefficient" -> 0.66,
    "SpecularAnisotropyCoefficient" -> 0.8,
    "Reflectance" -> 0.5,
    "CoatColor" -> {1.0, 1.0, 1.0},
    "CoatRoughnessCoefficient" -> 0.0,
    "CoatAnisotropyCoefficient" -> 0.0,
    "CoatReflectance" -> 0.5,
    "SpecularColorMultiplier" -> 1.0,
    "DiffuseColorMultiplier" -> 1.0,
    "CoatColorMultiplier" -> 0.0
|>;


(* ::Subsubsection::Closed:: *)
(*Base materials*)


$baseMaterials = Keys[materialParameters];


(* ::Subsection::Closed:: *)
(*messageRenderFunction*)


Options[messageRenderFunction] = Options[OpenVDBLevelSetRender];


messageRenderFunction[head_, expr_, ___] /; messageScalarGridQ[expr, head] = $Failed;


messageRenderFunction[head_, expr_, ___] /; messageLevelSetGridQ[expr, head] = $Failed;


messageRenderFunction[head_, vdb_, opts:OptionsPattern[]] := messageRenderFunction[head, vdb, Automatic, opts]


messageRenderFunction[head_, vdb_, shading_, opts:OptionsPattern[]] /; !OptionQ[shading] :=
    Block[{assoc, opt},
        assoc = iparseRenderOptions[vdb, shading, opts];
        (
            opt = renderFailureOption[assoc];
            (
                If[opt === "Shader",
                    Message[head::shaderval, shading],
                    Message[head::renderval, opt -> OptionValue[opt]]
                ];
                $Failed

            ) /; opt =!= $Failed

        ) /; AssociationQ[assoc]
    ]


messageRenderFunction[___] = $Failed


General::renderval = "`1` is an invalid render setting.";
General::shaderval = "`1` is an invalid shader setting.";
