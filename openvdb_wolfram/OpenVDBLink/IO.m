(* ::Package:: *)

(* ::Title:: *)
(*IO*)


(* ::Subtitle:: *)
(*Import & export vdb.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBImport"]
PackageExport["OpenVDBExport"]


OpenVDBImport::usage = "OpenVDBImport[\"file.vdb\"] imports an OpenVDB grid.";
OpenVDBExport::usage = "OpenVDBExport[\"file.vdb\", expr] exports data from an OpenVDB grid into a file.";


(* ::Section:: *)
(*VDB*)


(* ::Subsection::Closed:: *)
(*OpenVDBImport*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBImport[args___] /; !CheckArgs[OpenVDBImport[args], {1, 3}] = $Failed;


OpenVDBImport[args___] :=
    With[{res = iOpenVDBImport[args]},
        res /; res =!= $Failed
    ]


OpenVDBImport[args___] := mOpenVDBImport[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBImport*)


iOpenVDBImport[File[file_String], args___] := iOpenVDBImport[file, args]


iOpenVDBImport[file_String] := iOpenVDBImport[file, Automatic]


iOpenVDBImport[url_?URLStringQ, args___] :=
    Block[{extension, tempfile, dl, res},
        extension = urlExtension[url];
        tempfile = FileNameJoin[{$TemporaryDirectory, "temp." <> extension}];
        If[FileExistsQ[tempfile],
            Quiet[DeleteFile[tempfile]];
        ];

        dl = Quiet[URLDownload[url, tempfile]];
        (
            res = iOpenVDBImport[tempfile, args];
            Quiet[DeleteFile[tempfile]];

            res /; OpenVDBGridQ[res]

        ) /; FileExistsQ[tempfile]
    ]


iOpenVDBImport[file_?zipFileQ, args___] :=
    Block[{vdbfile, res},
        vdbfile = extractVDBZIP[file];
        (
            res = iOpenVDBImport[vdbfile, args];
            Quiet[DeleteFile[vdbfile]];

            res /; OpenVDBGridQ[res]

        ) /; FileExistsQ[vdbfile]
    ]


iOpenVDBImport[file_String?FileExistsQ, iname_, itype_:Automatic] :=
    Block[{name, type, vdb, id, successQ},
        name = If[StringQ[iname], iname, ""];
        type = If[itype === Automatic, detectVDBType[file, name], itype];
        (
            vdb = OpenVDBCreateGrid[1.0, type];
            (
                successQ = vdb["importVDB"[file, name]];

                vdb /; successQ

            ) /; OpenVDBGridQ[vdb]

        ) /; type =!= $Failed
    ]


iOpenVDBImport[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBImport] = {"ArgumentsPattern" -> {_, _., _.}};


addCodeCompletion[OpenVDBImport][None, None, $gridTypeList];


(* ::Subsubsection::Closed:: *)
(*Utilities*)


detectVDBType[file_, name_] :=
    Block[{vdb, type, wltype},
        (* create a grid of any type since it has the base methods *)
        vdb = OpenVDBCreateGrid[1.0, "Scalar"];

        type = vdb["importVDBType"[file, name]];
        (
            wltype = fromInternalType[type];

            wltype /; StringQ[wltype]

        ) /; StringQ[type]
    ]


detectVDBType[___] = $Failed;


URLStringQ[url_String?StringQ] := StringMatchQ[url, "http://*" | "ftp://*" | "https://*"]


URLStringQ[___] = False;


urlExtension[url_] := Replace[StringCases[StringDelete[url, Longest["?" ~~ ___ ~~ EndOfString]], Shortest["." ~~ __ ~~ EndOfString]], {} -> {"vdb"}, 0][[1]]


urlExtension[___] = "vdb";


zipFileQ[file_String?StringQ] := ToLowerCase[FileExtension[file]] === "zip"
zipFileQ[___] = False;


vdbFileQ[file_String?StringQ] := ToLowerCase[FileExtension[file]] === "vdb"
vdbFileQ[___] = False;


extractVDBZIP[file_] :=
    Block[{files, vdbfiles},
        files = Quiet[Import[file, "FileNames"]];
        (
            vdbfiles = Quiet[ExtractArchive[file, $TemporaryDirectory]];

            vdbfiles[[1]] /; MatchQ[vdbfiles, {_?StringQ}] && FileExistsQ[vdbfiles[[1]]]

        ) /; MatchQ[files, {_?vdbFileQ}]
    ]


extractVDBZIP[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBImport[expr_, ___] /; !FileExistsQ[expr] :=
    (
        Message[OpenVDBImport::nffil, expr, Import];
        $Failed
    )


mOpenVDBImport[expr_, _, type_, ___] /; type =!= Automatic && !MemberQ[$gridTypeList, type] :=
    (
        Message[OpenVDBImport::type, type];
        $Failed
    )


mOpenVDBImport[___] = $Failed;


OpenVDBImport::nffil = "File `1` not found during `2`.";


OpenVDBImport::type = "`1` is not a supported grid type. Evaluate OpenVDBGridTypes[] to see the list of supported types."


(* ::Subsection::Closed:: *)
(*OpenVDBExport*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBExport] = {OverwriteTarget -> False};


OpenVDBExport[args___] /; !CheckArgs[OpenVDBExport[args], {1, 2}] = $Failed;


OpenVDBExport[args___] :=
    With[{res = iOpenVDBExport[args]},
        res /; res =!= $Failed
    ]


OpenVDBExport[args___] := mOpenVDBExport[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBExport*)


Options[iOpenVDBExport] = Options[OpenVDBExport];


iOpenVDBExport[File[file_String], args___] := iOpenVDBExport[file, args]


iOpenVDBExport[file_String, vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    If[fileExportQ[file, OptionValue[OverwriteTarget]],
        vdb["exportVDB"[file]];
        file,
        $Failed
    ]


iOpenVDBExport[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBExport, 2];


SyntaxInformation[OpenVDBExport] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*fileExportQ*)


fileExportQ[filename_, overwriteQ_] :=
    If[!FileExistsQ[filename],
        True,
        If[TrueQ[overwriteQ],
            True,
            Message[OpenVDBExport::filex, filename];
            False
        ]
     ];


OpenVDBExport::filex = "Cannot overwrite existing file `1`.";


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBExport[expr_, ___] /; !MatchQ[expr, File[_String]|_String] :=
    (
        Message[OpenVDBExport::chtype, expr];
        $Failed
    )


mOpenVDBExport[_, expr_, ___] /; messageGridQ[expr, OpenVDBExport] = $Failed;


mOpenVDBExport[___] = $Failed;


OpenVDBExport::chtype = "First argument `1` is not a valid file specification.";
