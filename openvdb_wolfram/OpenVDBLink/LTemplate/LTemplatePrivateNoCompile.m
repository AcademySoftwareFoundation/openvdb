(* Mathematica Package *)

(* :Package Version: 0.5.4 *)
(* :Copyright: (c) 2019 Szabolcs Horvat *)
(* :License: MIT license, see LICENSE.txt *)

(*
 * To include LTemplate privately in another package, and disable compilation support,
 * load LTemplatePrivateNoCompile.m using Get[], then immediately call ConfigureLTemplate[].
 * Disabling compilation support will avoid loading CCompilerDriver` and therefore improve
 * loading performance on Windows. Packages that ship with pre-compiled binaries do not need
 * compilation support in the LTemplate they embed.
 *)

BeginPackage["`LTemplate`", {"SymbolicC`"}]

(* Note: Do not Protect symbols when LTemplate is loaded privately. *)

`Private`$private = True;
`Private`$noCompile = True;
Quiet[
  Get@FileNameJoin[{DirectoryName[$InputFileName], "LTemplateInner.m"}],
  General::shdw (* suppress false shadowing warnings if public LTemplate was loaded first *)
]

EndPackage[]
