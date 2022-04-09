(* ::Package:: *)

(* ::Text:: *)
(*This is a traditional package that uses BeginPackage instead of Package.*)
(*It loads LTemplate into the correct context.*)


BeginPackage["OpenVDBLink`"]


Get["LTemplate`LTemplatePrivate`"];
ConfigureLTemplate["MessageSymbol" -> OpenVDBLink, "LazyLoading" -> False]


EndPackage[]
