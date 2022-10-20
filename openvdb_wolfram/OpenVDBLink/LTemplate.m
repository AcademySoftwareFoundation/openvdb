(* ::Package:: *)

(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Text:: *)
(*This is a traditional package that uses BeginPackage instead of Package.*)
(*It loads LTemplate into the correct context.*)


BeginPackage["OpenVDBLink`"]


Get["OpenVDBLink`LTemplate`LTemplatePrivate`"];
ConfigureLTemplate["MessageSymbol" -> OpenVDBLink, "LazyLoading" -> False]


EndPackage[]
