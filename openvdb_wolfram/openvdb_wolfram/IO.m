(* ::Package:: *)

(* ::Title:: *)
(*IO*)


(* ::Subtitle:: *)
(*Import & export vdb.*)


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


OpenVDBImport[File[file_String], args___] := OpenVDBImport[file, args]


OpenVDBImport[file_String] := OpenVDBImport[file, Automatic]


OpenVDBImport[file_String?FileExistsQ, iname_, itype_:Automatic] := 
	Block[{name, type, vdb, id, successQ},
		name = If[StringQ[iname], iname, ""];
		type = If[itype === Automatic, detectVDBType[file, name], itype];
		
		vdb = OpenVDBCreateGrid[1.0, type];
		(	
			successQ = vdb["importVDB"[file, name]];
			
			vdb /; successQ
			
		) /; OpenVDBGridQ[vdb]
	]


OpenVDBImport[___] = $Failed;


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


(* ::Subsection::Closed:: *)
(*OpenVDBExport*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBExport] = {OverwriteTarget -> False};


OpenVDBExport[File[file_String], args___] := OpenVDBExport[file, args]


OpenVDBExport[file_String, vdb_?OpenVDBGridQ] := 
	If[fileExportQ[file, OptionValue[OverwriteTarget]],		
		vdb["exportVDB"[file]];
		file,
		$Failed
	]


OpenVDBExport[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBExport, 2];


SyntaxInformation[OpenVDBExport] = {"ArgumentsPattern" -> {_, _}};


(* ::Subsubsection::Closed:: *)
(*fileExportQ*)


fileExportQ[filename_, overwriteQ_] := !FileExistsQ[filename] || TrueQ[overwriteQ];
