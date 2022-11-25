(* Mathematica Package *)

(* :Package Version: 0.5.4 *)
(* :Copyright: (c) 2019 Szabolcs Horvat *)
(* :License: MIT license, see LICENSE.txt *)

(*
 * To include LTemplate privately in another package, load LTemplatePrivate.m using Get[],
 * then immediately call ConfigureLTemplate[].
 *)

BeginPackage["`LTemplate`", {"SymbolicC`", "CCompilerDriver`"}]

(* Note: Do not Protect symbols when LTemplate is loaded privately. *)

`Private`$private = True;
Quiet[
  Get@FileNameJoin[{DirectoryName[$InputFileName], "LTemplateInner.m"}],
  General::shdw (* suppress false shadowing warnings if public LTemplate was loaded first *)
]

EndPackage[]
