(* Mathematica Package *)
(* Created by IntelliJ IDEA and http://wlplugin.halirutan.de/ *)

(* :Title:   LTemplate    *)
(* :Context: LTemplate`   *)
(* :Author:  szhorvat     *)
(* :Date:    2015-08-03   *)

(* :Package Version: 0.6 *)
(* :Mathematica Version: 10.0 *)
(* :Copyright: (c) 2019 Szabolcs Horvat *)
(* :License: MIT license, see LICENSE.txt *)
(* :Keywords: LibraryLink, C++, Template, Code generation *)
(* :Discussion: This package simplifies writing LibraryLink code by auto-generating the boilerplate code. *)

BeginPackage["LTemplate`", {"SymbolicC`", "CCompilerDriver`"}]

Unprotect["LTemplate`*"];

`Private`$private = False;
Get["LTemplate`LTemplateInner`"]

ConfigureLTemplate[] (* use the default configuration *)

EndPackage[]

(* Note: Take care not to introduce any new symbols in this section as they would be created in Global` *)

SetAttributes[Evaluate@Names["LTemplate`*"], {Protected, ReadProtected}];

(* Add the class context to $ContextPath *)
If[Not@MemberQ[$ContextPath, LClassContext[]],
  PrependTo[$ContextPath, LClassContext[]]
];
