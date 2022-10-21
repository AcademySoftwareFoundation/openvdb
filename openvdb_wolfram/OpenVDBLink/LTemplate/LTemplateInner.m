(* ::Package:: *)

(* Mathematica Package *)

(* :Copyright: (c) 2019 Szabolcs Horvat *)
(* :License: MIT license, see LICENSE.txt *)

(* This file is read directly with Get in LTemplate.m or LTemplatePrivate.m *)


LTemplate::usage = "LTemplate[name, {LClass[\[Ellipsis]], LClass[\[Ellipsis]], \[Ellipsis]}] represents a library template.";
LClass::usage = "LClass[name, {fun1, fun2, \[Ellipsis]}] represents a class within a template.";
LFun::usage =
    "LFun[name, {arg1, arg2, \[Ellipsis]}, ret] represents a class member function with the given name, argument types and return type.\n" <>
    "LFun[name, LinkObject, LinkObject] represents a function that uses MathLink/WSTP based passing. The shorthand LFun[name, LinkObject] can also be used.";

LType::usage =
     "LType[head] represents an array-like library type corresponding to head.\n" <>
     "LType[head, etype] represents an array-like library type corresponding to head, with element type etype.\n" <>
     "LType[head, etype, d] represents an array-like library type corresponding to head, with element type etype and depth/rank d.";

TranslateTemplate::usage = "TranslateTemplate[template] translates the template into C++ code.";

LoadTemplate::usage = "LoadTemplate[template] loads the library defined by the template. The library must already be compiled.";
UnloadTemplate::usage = "UnloadTemplate[template] attempts to unload the library defined by the template.";

CompileTemplate::usage =
    "CompileTemplate[template] compiles the library defined by the template. Required source files must be present in the current directory.\n" <>
    "CompileTemplate[template, {file1, \[Ellipsis]}] includes additional source files in the compilation.";

FormatTemplate::usage = "FormatTemplate[template] formats the template in an easy to read way.";

NormalizeTemplate::usage = "NormalizeTemplate[template] brings the template and the type specifications within to the canonical form used internally by other LTemplate functions.";

ValidTemplateQ::usage = "ValidTemplateQ[template] returns True if the template syntax is valid.";

Make::usage = "Make[class] creates an instance of class.";

LExpressionList::usage = "LExpressionList[class] returns all existing instances of class.";

LClassContext::usage = "LClassContext[] returns the context where class symbols are created.";

LExpressionID::usage = "LExpressionID[name] represents the data type corresponding to LClass[name, \[Ellipsis]] in templates.";

ConfigureLTemplate::usage = "ConfigureLTemplate[options] must be called after loading the LTemplate package privately.";

Begin["`Private`"] (* Begin Private Context *)

(* Private for now, use LFun[name, LinkObject, LinkObject] instead. *)
LOFun::usage =
    "LOFun[name] represents a class member function that uses LinkObject for passing and returning arguments. " <>
    "It is equivalent to LFun[name, LinkObject, LinkObject].";

(* Mathematica version checks *)

packageAbort[] := (End[]; EndPackage[]; Abort[]) (* Avoid polluting the context path when aborting early. *)

minVersion = {10.0, 0}; (* oldest supported Mathematica version *)
maxVersion = {12.1, 1}; (* latest Mathematica version the package was tested with *)
version    = {$VersionNumber, $ReleaseNumber}
versionString[{major_, release_}] := StringJoin[ToString /@ {NumberForm[major, {Infinity, 1}], ".", release}]

If[Not@OrderedQ[{minVersion, version}],
  Print["LTemplate requires at least Mathematica version " <> versionString[minVersion] <> ".  Aborting."];
  packageAbort[];
]

(* We need to rely on implementation details of SymbolicC, so warn users of yet untested new Mathematica versions. *)
(*If[Not@OrderedQ[{version, maxVersion}] && Not[$private],
  Print[
    StringTemplate[
      "WARNING: LTemplate has not yet been tested with Mathematica ``.\n" <>
      "The latest supported Mathematica version is ``.\n" <>
      "Please report any issues you find to szhorvat at gmail.com."
    ][versionString[version], versionString[maxVersion]]
  ]
]*)

(*************** Package configuration ****************)

(* Set up package global variables *)

$packageDirectory = DirectoryName[$InputFileName];
$includeDirectory = FileNameJoin[{$packageDirectory, "IncludeFiles"}];

(* The following symbols are set by ConfigureLTemplate[] *)
$messageSymbol := warnConfig
$lazyLoading := warnConfig

(* Show error and abort when ConfigureLTemplate[] was not called. *)
warnConfig := (Print["FATAL ERROR: Must call ConfigureLTemplate[] when embedding LTemplate into another package. Aborting ..."]; Abort[])

ByteArray[{0}]; (* "Prime" ByteArray to work around 10.4 bug where returning ByteArrays works only after they have been used once *)

LibraryFunction::noinst = "Managed library expression instance does not exist.";

LTemplate::nofun = "Function `` does not exist.";


Options[ConfigureLTemplate] = { "MessageSymbol" -> LTemplate, "LazyLoading" -> False };

ConfigureLTemplate[opt : OptionsPattern[]] :=
    With[{sym = OptionValue["MessageSymbol"]},
      $messageSymbol = sym;
      sym::info    = "``";
      sym::warning = "``";
      sym::error   = "``";
      sym::assert  = "Assertion failed: ``.";

      $lazyLoading = OptionValue["LazyLoading"];
    ]

LClassContext[] = Context[LTemplate] <> "Classes`";


(***************** SymbolicC extensions *******************)

CDeclareAssign::usage = "CDeclareAssign[type, var, value] represents 'type var = value;'.";
SymbolicC`Private`IsCExpression[ _CDeclareAssign ] := True
GenerateCode[CDeclareAssign[typeArg_, idArg_, rhs_], opts : OptionsPattern[]] :=
    Module[{type, id},
      type = Flatten[{typeArg}];
      id = Flatten[{idArg}];
      type = Riffle[ Map[ GenerateCode[#, opts] &, type], " "];
      id = Riffle[ Map[ GenerateCode[#, opts] &, id], ", "];
      GenerateCode[CAssign[type <> " " <> id, rhs], opts]
    ]

CInlineCode::usage = "CInlineCode[\"some code\"] will prevent semicolons from being added at the end of \"some code\" when used in a list.";
GenerateCode[CInlineCode[arg_], opts : OptionsPattern[]] := GenerateCode[arg, opts]

CTryCatch::usage = "CTryCatch[tryCode, catchArg, catchCode] represents 'try { tryCode } catch (catchArg) { catchCode }'.";
GenerateCode[CTryCatch[try_, arg_, catch_], opts : OptionsPattern[]] :=
    GenerateCode[CTry[try], opts] <> "\n" <> GenerateCode[CCatch[arg, catch], opts]

CTry::usage = "CTry[tryCode] represents the fragment 'try { tryCode }'. Use CTryCatch instead.";
GenerateCode[CTry[try_], opts : OptionsPattern[]] :=
    "try\n" <> GenerateCode[CBlock[try], opts]

CCatch::usage = "CCatch[catchArg, catchCode] represents the fragment 'catch (catchArg) { catchCode }'. Use CTryCatch instead.";
GenerateCode[CCatch[arg_, catch_], opts : OptionsPattern[]] :=
    "catch (" <> SymbolicC`Private`formatArgument[arg, opts] <> ")\n" <>
        GenerateCode[CBlock[catch], opts]

(****************** Generic template processing ****************)

numericTypePattern = Integer|Real|Complex;
rawTypePattern     = "Integer8"|"UnsignedInteger8"|"Integer16"|"UnsignedInteger16"|"Integer32"|"UnsignedInteger32"|
                     "Integer64"|"UnsignedInteger64"|"Real32"|"Real64"|"Complex64"|"Complex128";
imageTypePattern   = "Bit"|"Byte"|"Bit16"|"Real32"|"Real";

passingMethodPattern = PatternSequence[]|"Shared"|"Manual"|"Constant"|Automatic;

depthPattern = _Integer?Positive | Verbatim[_];
depthNullPattern = PatternSequence[] | depthPattern; (* like depthPattern, but allow empty value*)

arrayPattern = LType[List, numericTypePattern, depthNullPattern]; (* disallow MTensor without explicit element type specification *)
sparseArrayPattern = LType[SparseArray, numericTypePattern, depthNullPattern]; (* disallow SparseArray without explicit element type specification *)
rawArrayPattern = LType[RawArray, rawTypePattern] | LType[RawArray];
numericArrayPattern = LType[NumericArray, rawTypePattern] | LType[NumericArray]
byteArrayPattern = LType[ByteArray];
imagePattern = LType[Image|Image3D, imageTypePattern] | LType[Image|Image3D];

(*
  Normalizing a template will:
  - Wrap a bare LClass with LTemplate.  This way a bare LClass can be used as a shorter notation for a single-class template.
  - Convert type names to a canonical form
  - Convert LFun[name, LinkObject, LinkObject] to LOFun[name]
*)

normalizeTypesRules = Dispatch@{
  (* convert pattern-like type specifications to type names *)
  Verbatim[_Integer] -> Integer,
  Verbatim[_Real] -> Real,
  Verbatim[_Complex] -> Complex,
  Verbatim[True|False] -> "Boolean",
  Verbatim[False|True] -> "Boolean",

  (* convert string heads to symbols *)
  (* must only be used on type lists as an LTemplate expression may contain other strings *)
  head : "List"|"SparseArray"|"Image"|"Image3D"|"RawArray"|"NumericArray"|"ByteArray" :> Symbol[head],

  (* convert LibraryDataType to the more general LType *)
  LibraryDataType[args__] :> LType[args]
};

(* These heads are allowed to appear on their own, without being wrapped in LType/LibraryDataType.
   This is for consistency with plain LibraryLink. *)
nakedHeads = ByteArray|RawArray|NumericArray|Image|Image3D;
wrapNakedHeadsRules = Dispatch@{
  expr : LType[___] :> expr, (* do not wrap if already wrapped *)
  type : nakedHeads :> LType[type]
};

elemTypeAliases = Dispatch@{
  LType[h: RawArray|NumericArray, "Byte"]    :> LType[h, "UnsignedInteger8"],
  LType[h: RawArray|NumericArray, "Bit16"]   :> LType[h, "UnsignedInteger16"],
  (* omit "Integer" because the naming is confusing and people may assume it's "Integer64" *)
  (* LType[h: RawArray|NumericArray, "Integer"]          :> LType[h, "Integer32"], *)
  LType[h: RawArray|NumericArray, "Float"]   :> LType[h, "Real32"],
  LType[h: RawArray|NumericArray, "Double"]  :> LType[h, "Real64"],
  LType[h: RawArray|NumericArray, "Real"]    :> LType[h, "Real64"],
  LType[h: RawArray|NumericArray, "Complex"] :> LType[h, "Complex128"],

  LType[h : Image|Image3D, "UnsignedInteger8"]  :> LType[h, "Byte"],
  LType[h : Image|Image3D, "UnsignedInteger16"] :> LType[h, "Bit16"],
  LType[h : Image|Image3D, "Float"]             :> LType[h, "Real32"],
  LType[h : Image|Image3D, "Double"]            :> LType[h, "Real"],
  LType[h : Image|Image3D, "Real64"]            :> LType[h, "Real"]
};

normalizeFunsRules = Dispatch@{
  LFun[name_, LinkObject] :> LOFun[name],
  LFun[name_, LinkObject, LinkObject] :> LOFun[name],
  LFun[name_, args_List, ret_] :> LFun[name, normalizeTypes[args, 1], normalizeTypes[ret]]
};

(* These rules must only be applied to entire type specifications, not their parts. Use Replace, not ReplaceAll. *)
typeRules = Dispatch@{
  (* allowed forms of tensor specifications include {type}, {type, depth}, {type, depth, passing}, but NOT {type, passing} *)
  {type : numericTypePattern, depth : depthPattern, pass : passingMethodPattern} :> {LType[List, type, depth], pass},
  {type : numericTypePattern, pass : passingMethodPattern} :> {LType[List, type], pass},
  type : LType[__] :> {type}
};

normalizeTypes[types_, level_ : 0] := Replace[types /. normalizeTypesRules /. wrapNakedHeadsRules /. elemTypeAliases, typeRules, {level}]

NormalizeTemplate[c : LClass[name_, funs_]] := NormalizeTemplate[LTemplate[name, {c}]]
NormalizeTemplate[t : LTemplate[name_, classes_]] := t /. normalizeFunsRules
NormalizeTemplate[t_] := t


ValidTemplateQ::template = "`` is not a valid template. Templates must follow the syntax LTemplate[name, {class1, class2, \[Ellipsis]}].";
ValidTemplateQ::class    = "In ``: `` is not a valid class. Classes must follow the syntax LClass[name, {fun1, fun2, \[Ellipsis]}].";
ValidTemplateQ::fun      = "In ``: `` is not a valid function. Functions must follow the syntax LFun[name, {arg1, arg2, \[Ellipsis]}, ret].";
ValidTemplateQ::string   = "In ``: String expected instead of ``";
ValidTemplateQ::name     = "In ``: `` is not a valid name. Names must start with a letter and may only contain letters and digits.";
ValidTemplateQ::type     = "In ``: `` is not a valid type.";
ValidTemplateQ::rettype  = "In ``: `` is not a valid return type.";
ValidTemplateQ::dupclass = "In ``: Class `` appears more than once.";
ValidTemplateQ::dupfun   = "In ``: Function `` appears more than once.";

ValidTemplateQ[tem_] := validateTemplate@NormalizeTemplate[tem]


validateTemplate[tem_] := (Message[ValidTemplateQ::template, tem]; False)
validateTemplate[LTemplate[name_String, classes_List]] :=
    Block[{classlist = {}, location = "template"},
      validateName[name] && (And @@ validateClass /@ classes)
    ]

(* must be called within validateTemplate, uses location, classlist *)
validateClass[class_] := (Message[ValidTemplateQ::class, location, class]; False)
validateClass[LClass[name_, funs_List]] :=
    Block[{funlist = {}, inclass, nameValid},
      nameValid = validateName[name];
      If[MemberQ[classlist, name], Message[ValidTemplateQ::dupclass, location, name]; Return[False]];
      AppendTo[classlist, name];
      inclass = name;
      Block[{location = StringTemplate["class ``"][inclass]},
        nameValid && (And @@ validateFun /@ funs)
      ]
    ]

(* must be called within validateClass, uses location, funlist *)
validateFun[fun_] := (Message[ValidTemplateQ::fun, location, fun]; False)
validateFun[LFun[name_, args_List, ret_]] :=
    Block[{nameValid},
      nameValid = validateName[name];
      If[MemberQ[funlist, name], Message[ValidTemplateQ::dupfun, location, name]; Return[False]];
      AppendTo[funlist, name];
      Block[{location = StringTemplate["class ``, function ``"][inclass, name]},
        nameValid && (And @@ validateType /@ args) && validateReturnType[ret]
      ]
    ]
validateFun[LOFun[name_]] :=
    Block[{nameValid},
      nameValid = validateName[name];
      If[MemberQ[funlist, name], Message[ValidTemplateQ::dupfun, location, name]; Return[False]];
      AppendTo[funlist, name];
      nameValid
    ]

(* must be called within validateTemplate, uses location *)
validateType[numericTypePattern|"Boolean"|"UTF8String"|LExpressionID[_String]] := True
validateType[{arrayPattern|sparseArrayPattern|rawArrayPattern|numericArrayPattern|byteArrayPattern|imagePattern, passingMethodPattern}] := True
validateType[type_] := (Message[ValidTemplateQ::type, location, type]; False)

(* must be called within validateTemplate, uses location *)
(* Only "Shared" and Automatic passing allowed in return types. LExpressionID is forbidden. *)
validateReturnType["Void"] := True
validateReturnType[type : LExpressionID[___] | {___, "Manual"|"Constant"}] := (Message[ValidTemplateQ::rettype, location, type]; False)
validateReturnType[type_] := validateType[type]

(* must be called within validateTemplate, uses location *)
validateName[name_] := (Message[ValidTemplateQ::string, location, name]; False)
validateName[name_String] :=
    If[StringMatchQ[name, RegularExpression["[a-zA-Z][a-zA-Z0-9]*"]],
      True,
      Message[ValidTemplateQ::name, location, name]; False
    ]



$indent = "    ";


(***********  Translate template to library code  **********)

TranslateTemplate[tem_] :=
    With[{t = NormalizeTemplate[tem]},
      If[validateTemplate[t],
        StringReplace[ToCCodeString[transTemplate[t], "Indent" -> $indent], Longest[" "..] ~~ "\n" -> "\n"],
        $Failed
      ]
    ]


libFunArgs = {{"WolframLibraryData", "libData"}, {"mint", "Argc"}, {"MArgument *", "Args"}, {"MArgument", "Res"}};
linkFunArgs = {{"WolframLibraryData", "libData"}, {"MLINK", "mlp"}};
libFunRet  = "extern \"C\" DLLEXPORT int";

excType = "const mma::LibraryError &";
excName = "libErr";

varPrefix = "var";
var[k_] := varPrefix <> IntegerString[k]

includeName[classname_String] := classname <> ".h"

collectionName[classname_String] := classname <> "_collection"

collectionType[classname_String] := "std::map<mint, " <> classname <> " *>"

managerName[classname_String] := classname <> "_manager_fun"

fullyQualifiedSymbolName[sym_Symbol] := Context[sym] <> SymbolName[sym]


setupCollection[classname_String] := {
  CDeclare[collectionType[classname], collectionName[classname]],
  "",
  CInlineCode["namespace mma"], (* workaround for gcc bug, "specialization of template in different namespace" *)
  CBlock@CFunction["template<> const " <> collectionType[classname] <> " &", "getCollection<" <> classname <> ">", {},
    CReturn[collectionName[classname]]
  ],
  "",
  CFunction["void", managerName[classname], {"WolframLibraryData libData", "mbool mode", "mint id"},
    CInlineCode@StringTemplate[ (* TODO: Check if id exists, use assert *)
      "\
if (mode == 0) { // create
`indent``collection`[id] = new `class`();
} else {  // destroy
`indent`if (`collection`.find(id) == `collection`.end()) {
`indent``indent`libData->Message(\"noinst\");
`indent``indent`return;
`indent`}
`indent`delete `collection`[id];
`indent``collection`.erase(id);
}\
"][<|"collection" -> collectionName[classname], "class" -> classname, "indent" -> $indent|>]
  ],
  "",
  CFunction[libFunRet, classname <> "_get_collection", libFunArgs,
    {
    (* Attention: make sure stuff called here won't throw LibraryError *)
      transRet[
        {LType[List, Integer, 1]},
        CCall["mma::detail::get_collection", collectionName[classname]]
      ],
      CReturn["LIBRARY_NO_ERROR"]
    }
  ],
  "",""
}


registerClassManager[classname_String] :=
    CBlock[{
      "int err",
      StringTemplate[
        "err = (*libData->registerLibraryExpressionManager)(\"`class`\", `manager`)"
      ][<|"class" -> classname, "manager" -> managerName[classname]|>],
      "if (err != LIBRARY_NO_ERROR) return err"
    }]


unregisterClassManager[classname_String] :=
    StringTemplate["(*libData->unregisterLibraryExpressionManager)(\"``\")"][classname]


transTemplate[LTemplate[libname_String, classes_]] :=
    Block[{classlist = {}, classTranslations},
      classTranslations = transClass /@ classes;
      {
        CComment["This file was automatically generated by LTemplate. DO NOT EDIT.", {"", "\n"}],
        CComment["https://github.com/szhorvat/LTemplate", {"", "\n"}],
        "",
        CDefine["LTEMPLATE_MMA_VERSION", ToString@Round[100 $VersionNumber + $ReleaseNumber]],
        "",
        CInclude["LTemplate.h"],
        CInclude["LTemplateHelpers.h"],
        CInclude /@ includeName /@ classlist,
        "","",

        CDefine["LTEMPLATE_MESSAGE_SYMBOL", CString[fullyQualifiedSymbolName[$messageSymbol]]],
        "",
        CInclude["LTemplate.inc"],

        "","",

        setupCollection /@ classlist,

        CFunction["extern \"C\" DLLEXPORT mint",
          "WolframLibrary_getVersion", {},
          "return WolframLibraryVersion"
        ],
        "",
        CFunction["extern \"C\" DLLEXPORT int",
          "WolframLibrary_initialize", {"WolframLibraryData libData"},
          {
            CAssign["mma::libData", "libData"],
            registerClassManager /@ classlist,
            "return LIBRARY_NO_ERROR"
          }
        ],
        "",
        CFunction["extern \"C\" DLLEXPORT void",
          "WolframLibrary_uninitialize", {"WolframLibraryData libData"},
          {
            unregisterClassManager /@ classlist,
            "return"
          }
        ],
        "","",
        classTranslations
      }
    ]


(* must be called within transTemplate *)
transClass[LClass[classname_String, funs_]] :=
    Block[{},
      AppendTo[classlist, classname];
      transFun[classname] /@ funs
    ]


funName[classname_][name_] := classname <> "_" <> name


catchExceptions[classname_, funname_] :=
    Module[{membername = "\"" <> classname <> "::" <> funname <> "()\""},
      {
        CCatch[{excType, excName},
          {
            CMember[excName, "report()"],
            CReturn[CMember[excName, "error_code()"]]
          }
        ]
        ,
        CCatch[
          {"const std::exception &", "exc"},
          {
            CCall["mma::detail::handleUnknownException", {"exc.what()", membername}],
            CReturn["LIBRARY_FUNCTION_ERROR"]
          }
        ]
        ,
        CCatch["...",
          {
            CCall["mma::detail::handleUnknownException", {"NULL", membername}],
            CReturn["LIBRARY_FUNCTION_ERROR"]
          }
        ]
      }
    ]


transFun[classname_][LFun[name_String, args_List, ret_]] :=
    Block[{index = 0},
      {
        CFunction[libFunRet, funName[classname][name], libFunArgs,
          {
            CDeclare["mma::detail::MOutFlushGuard", "flushguard"],
            CInlineCode@"if (setjmp(mma::detail::jmpbuf)) { return LIBRARY_FUNCTION_ERROR; }",
            (* TODO: check Argc is correct, use assert *)
            "const mint id = MArgument_getInteger(Args[0])",
            CInlineCode@StringTemplate[
              "if (`1`.find(id) == `1`.end()) { libData->Message(\"noinst\"); return LIBRARY_FUNCTION_ERROR; }"
            ][collectionName[classname]],
            "",
            CTry[
            (* try *) {
              transArg /@ args,
              "",
              transRet[
                ret,
                CPointerMember[CArray[collectionName[classname], "id"], CCall[name, var /@ Range@Length[args]]]
              ]
            }],
            (* catch *)
            catchExceptions[classname, name],
            "",
            CReturn["LIBRARY_NO_ERROR"]
          }
        ],
        "", ""
      }
    ]

transFun[classname_][LOFun[name_String]] :=
    {
      CFunction[libFunRet, funName[classname][name], linkFunArgs,
        {
          CDeclare["mma::detail::MOutFlushGuard", "flushguard"],
          CInlineCode@"if (setjmp(mma::detail::jmpbuf)) { return LIBRARY_FUNCTION_ERROR; }",
          CTry[
            (* try *) {
              CInlineCode@StringTemplate[
"
int id;
int args = 2;

if (! MLTestHeadWithArgCount(mlp, \"List\", &args))
`indent`return LIBRARY_FUNCTION_ERROR;
if (! MLGetInteger(mlp, &id))
`indent`return LIBRARY_FUNCTION_ERROR;
if (`collection`.find(id) == `collection`.end()) {
`indent`libData->Message(\"noinst\");
`indent`return LIBRARY_FUNCTION_ERROR;
}
`collection`[id]->`funname`(mlp);
"
              ][<| "collection" -> collectionName[classname], "funname" -> name, "indent" -> $indent |>]
            }
          ],
          (* catch *)
          catchExceptions[classname, name],
          "",
          CReturn["LIBRARY_NO_ERROR"]
        }
      ]
    }

transArg[type_] :=
    Module[{name, cpptype, getfun, setfun},
      index++;
      name = var[index];
      {cpptype, getfun, setfun} = Replace[type, types];
      {
        CDeclareAssign[cpptype, name, StringTemplate["`1`(Args[`2`])"][getfun, index]]
      }
    ]

transRet[type_, value_] :=
    Module[{name = "res", cpptype, getfun, setfun},
      {cpptype, getfun, setfun} = Replace[type, types];
      {
        CDeclareAssign[cpptype, name, value],
        CCall[setfun, {"Res", name}]
      }
    ]

transRet["Void", value_] := value


numericTypes = <|
  Integer -> "mint",
  Real    -> "double",
  Complex -> "mma::complex_t"
|>;

rawTypes = <|
  "Integer8"          -> "int8_t",
  "UnsignedInteger8"  -> "uint8_t",
  "Integer16"         -> "int16_t",
  "UnsignedInteger16" -> "uint16_t",
  "Integer32"         -> "int32_t",
  "UnsignedInteger32" -> "uint32_t",
  "Integer64"         -> "int64_t",
  "UnsignedInteger64" -> "uint64_t",
  "Real32"            -> "float",
  "Real64"            -> "double",
  "Complex32"         -> "mma::complex_float_t",
  "Complex64"         -> "mma::complex_double_t"
|>;

imageTypes = <|
  "Bit"    -> "mma::im_bit_t",
  "Byte"   -> "mma::im_byte_t",
  "Bit16"  -> "mma::im_bit16_t",
  "Real32" -> "mma::im_real32_t",
  "Real"   -> "mma::im_real_t"
|>;


types = Dispatch@{
  Integer      -> {"mint",                 "MArgument_getInteger",     "MArgument_setInteger"},
  Real         -> {"double",               "MArgument_getReal",        "MArgument_setReal"},
  Complex      -> {"std::complex<double>", "mma::detail::getComplex",  "mma::detail::setComplex"},
  "Boolean"    -> {"bool",                 "MArgument_getBoolean",     "MArgument_setBoolean"},
  "UTF8String" -> {"const char *",         "mma::detail::getString",   "mma::detail::setString"},

  {LType[List, type_, ___], ___} :>
      With[{ctype = numericTypes[type]},
        {"mma::TensorRef<" <> ctype <> ">", "mma::detail::getTensor<" <> ctype <> ">", "mma::detail::setTensor<" <> ctype <> ">"}
      ],

  {LType[SparseArray, type_, ___], ___} :>
      With[{ctype = numericTypes[type]},
        {"mma::SparseArrayRef<" <> ctype <> ">", "mma::detail::getSparseArray<" <> ctype <> ">", "mma::detail::setSparseArray<" <> ctype <> ">"}
      ],

  {LType[RawArray, type_], ___} :>
      With[
        {ctype = rawTypes[type]},
        {"mma::RawArrayRef<" <> ctype <> ">", "mma::detail::getRawArray<" <> ctype <> ">", "mma::detail::setRawArray<" <> ctype <> ">"}
      ],

  {LType[RawArray], ___} -> {"mma::GenericRawArrayRef", "mma::detail::getGenericRawArray", "mma::detail::setGenericRawArray"},

  {LType[NumericArray, type_], ___} :>
      With[
        {ctype = rawTypes[type]},
        {"mma::NumericArrayRef<" <> ctype <> ">", "mma::detail::getNumericArray<" <> ctype <> ">", "mma::detail::setNumericArray<" <> ctype <> ">"}
      ],

  {LType[NumericArray], ___} -> {"mma::GenericNumericArrayRef", "mma::detail::getGenericNumericArray", "mma::detail::setGenericNumericArray"},

  (* Starting with LTemplate 0.6, ByteArray is mapped to NumericArrayRef instead of RawArrayRef *)
  (* {LType[ByteArray], ___} -> {"mma::RawArrayRef<uint8_t>", "mma::detail::getRawArray<uint8_t>", "mma::detail::setRawArray<uint8_t>"}, *)
  {LType[ByteArray], ___} -> {"mma::NumericArrayRef<uint8_t>", "mma::detail::getNumericArray<uint8_t>", "mma::detail::setNumericArray<uint8_t>"},

  {LType[Image, type_], ___} :>
      With[
        {ctype = imageTypes[type]},
        {"mma::ImageRef<" <> ctype <> ">", "mma::detail::getImage<" <> ctype <> ">", "mma::detail::setImage<" <> ctype <> ">"}
      ],

  {LType[Image], ___} -> {"mma::GenericImageRef", "mma::detail::getGenericImage", "mma::detail::setGenericImage"},

  {LType[Image3D, type_], ___} :>
      With[
        {ctype = imageTypes[type]},
        {"mma::Image3DRef<" <> ctype <> ">", "mma::detail::getImage3D<" <> ctype <> ">", "mma::detail::setImage3D<" <> ctype <> ">"}
      ],

  {LType[Image3D], ___} -> {"mma::GenericImage3DRef", "mma::detail::getGenericImage3D", "mma::detail::setGenericImage3D"},

  (* This is a special type that translates integer managed expression IDs on the Mathematica side
     into a class reference on the C++ side. It cannot be returned. *)
  LExpressionID[classname_String] :> {classname <> " &", "mma::detail::getObject<" <> classname <> ">(" <> collectionName[classname]  <> ")", ""}
};



(**************** Load library ***************)

(* TODO: Break out loading and compilation into separate files
   This is to make it easy to include them in other projects *)

getCollection (* underlies LExpressionList, the get_collection library function is associated with it in loadClass *)

symName[classname_String] := LClassContext[] <> classname


LoadTemplate[tem_] :=
    With[{t = NormalizeTemplate[tem]},
      If[validateTemplate[t],
        Check[loadTemplate[t], $Failed],
        $Failed
      ]
    ]

(* We use FindLibrary for two reasons:
   1. If the library is not found, we want to fail early with LibraryFunction::notfound
   2. It is important to pass the full library path to LibraryFunctionLoad[]. If only a simple name is passed,
      it will use FindLibrary[] to find the appropriate file. FindLibrary[] is several orders of magnitude slower than just
      loading a function from a shared library. In fact, with this optimization, lazy loading might be pointless, as with
      full library paths, LibraryFunctionLoad[] is faster than other operations done during template loading.
 *)
loadTemplate[tem : LTemplate[libname_String, classes_]] :=
    With[{lib = FindLibrary[libname]},
      Quiet@unloadTemplate[tem];
      If[lib =!= $Failed,
        loadClass[lib] /@ classes,
        Message[LibraryFunction::notfound, libname]
      ];
    ]

loadClass[libname_][tem : LClass[classname_String, funs_]] := (
  ClearAll[#]& @ symName[classname];
  loadFun[libname, classname] /@ funs;
  With[{sym = Symbol@symName[classname]},
    MessageName[sym, "usage"] = formatTemplate[tem];
    sym[id_Integer][(f_String)[___]] /; (Message[LTemplate::nofun, StringTemplate["``::``"][sym, f]]; False) := $Failed;
    getCollection[sym] = LibraryFunctionLoad[libname, funName[classname]["get_collection"], {}, LibraryDataType[List, Integer, 1]];
  ];
)


loadFun[libname_, classname_][LFun[name_String, args_List, ret_]] :=
    With[{classsym = Symbol@symName[classname], funname = funName[classname][name],
      loadargs = Prepend[Replace[args, loadingTypes, {1}], Integer],
      loadret = Replace[ret, loadingTypes]
    },
      If[$lazyLoading,
        classsym[idx_Integer]@name[argumentsx___] :=
            With[{lfun = LibraryFunctionLoad[libname, funname, loadargs, loadret]},
              classsym[id_Integer]@name[arguments___] := lfun[id, arguments];
              classsym[idx]@name[argumentsx]
            ]
        ,
        With[{lfun = LibraryFunctionLoad[libname, funname, loadargs, loadret]},
          classsym[id_Integer]@name[arguments___] := lfun[id, arguments];
        ]
      ]
    ];

loadFun[libname_, classname_][LOFun[name_String]] :=
    With[{classsym = Symbol@symName[classname], funname = funName[classname][name]},
      If[$lazyLoading,
        classsym[idx_Integer]@name[argumentsx___] :=
          With[{lfun = LibraryFunctionLoad[libname, funname, LinkObject, LinkObject]},
            classsym[id_Integer]@name[arguments___] := lfun[id, {arguments}];
            classsym[idx]@name[argumentsx]
          ]
        ,
        With[{lfun = LibraryFunctionLoad[libname, funname, LinkObject, LinkObject]},
          classsym[id_Integer]@name[arguments___] := lfun[id, {arguments}];
        ]
      ]
    ]


(* For types that need to be translated to LibraryFunctionLoad compatible forms before loading. *)
loadingTypes = Dispatch@{
  LExpressionID[_] -> Integer,
  {LType[h: RawArray|NumericArray, ___], passing___} :> {h, passing},
  {LType[args__], passing___} :> {LibraryDataType[args], passing}
};


UnloadTemplate[tem_] :=
    With[{t = NormalizeTemplate[tem]},
      If[validateTemplate[t],
        unloadTemplate[t],
        $Failed
      ]
    ]

unloadTemplate[LTemplate[libname_String, classes_]] :=
    Module[{res},
      res = LibraryUnload[libname];
      With[{syms = Symbol /@ symName /@ Cases[classes, LClass[name_, __] :> name]},
        ClearAll /@ syms;
        Quiet@Unset[getCollection[#]]& /@ syms;
      ];
      res
    ]


(* TODO: verify class exists for Make and LExpressionList *)

Make[class_Symbol] := Make@SymbolName[class] (* SymbolName returns the name of the symbol without a context *)
Make[classname_String] := CreateManagedLibraryExpression[classname, Symbol@symName[classname]]


LExpressionList[class_Symbol] := class /@ getCollection[class][]
LExpressionList[classname_String] := LExpressionList@Symbol@symName[classname]


(********************* Compile template ********************)

CompileTemplate::comp = "The compiler specification `` is invalid. It must be a symbol.";

If[TrueQ[$noCompile],

  (* If CCompilerDriver has not been loaded: *)
  CompileTemplate::disabled = "Template compilation is disabled.";
  CompileTemplate[___] := (Message[CompileTemplate::disabled]; $Failed);
  ,

  (* If CCompilerDriver is available: *)
  CompileTemplate[tem_, sources_List, opt : OptionsPattern[CreateLibrary]] :=
      With[{t = NormalizeTemplate[tem]},
        If[validateTemplate[t],
          compileTemplate[t, sources, opt],
          $Failed
        ]
      ];
  CompileTemplate[tem_, opt : OptionsPattern[CreateLibrary]] := CompileTemplate[tem, {}, opt];
]

compileTemplate[tem: LTemplate[libname_String, classes_], sources_, opt : OptionsPattern[CreateLibrary]] :=
    Catch[
      Module[{sourcefile, code, includeDirs, classlist, print, driver},
        print[args__] := Apply[Print, Style[#, Darker@Blue]& /@ {args}];

        (* Determine the compiler driver that will be used. *)
        (* It is unclear if the "Compiler" option of CreateLibrary supports option lists as a compiler specification
           like $CCompiler does. Trying to use one frequently leads to errors as of M11.2.  This may or may not be a bug.
           For now we forbid anything but symbol compiler specifications, such as CCompilerDriver`ClangCompiler`ClangCompiler *)
        driver = OptionValue["Compiler"];
        If[driver === Automatic, driver = DefaultCCompiler[]];
        If[driver === $Failed, Throw[$Failed, compileTemplate]];
        If[Not@MatchQ[driver, _Symbol],
          Message[CompileTemplate::comp, driver];
          Throw[$Failed, compileTemplate]
        ];

        print["Current directory is: ", Directory[]];
        classlist = Cases[classes, LClass[s_String, __] :> s];
        sourcefile = "LTemplate-" <> libname <> ".cpp";
        If[Not@FileExistsQ[#],
          print["File ", #, " does not exist.  Aborting."]; Throw[$Failed, compileTemplate]
        ]& /@ (# <> ".h"&) /@ classlist;
        print["Unloading library ", libname, " ..."];
        Quiet@LibraryUnload[libname];
        print["Generating library code ..."];
        code = TranslateTemplate[tem];
        If[FileExistsQ[sourcefile], print[sourcefile, " already exists and will be overwritten."]];
        Export[sourcefile, code, "String"];
        print["Compiling library code ..."];
        includeDirs = Flatten[{OptionValue["IncludeDirectories"], $includeDirectory}];

        With[{driver = driver},
          Internal`InheritedBlock[{driver},
            SetOptions[driver,
              "SystemCompileOptions" -> Flatten@{
                OptionValue[driver, "SystemCompileOptions"],
                Switch[{$OperatingSystem, driver["Name"][]},
                  {"Windows", "Visual Studio"}, {},
                  {"Windows", "Intel Compiler"}, {},
                  {"MacOSX", "Clang"}, {},
                  {_, _}, {}
                ]
              }
            ];
            CreateLibrary[
              AbsoluteFileName /@ Flatten[{sourcefile, sources}], libname,
              "IncludeDirectories" -> includeDirs,
              Sequence @@ FilterRules[{opt}, Except["IncludeDirectories"]]
            ]
          ]
        ]
      ],
      compileTemplate
    ]


(****************** Pretty print a template ********************)

FormatTemplate[template_] :=
    With[{t = NormalizeTemplate[template]},
      (* If the template is invalid, we report errors but we do not abort.
         Pretty-printing is still useful for invalid templates to facilitate finding mistakes.
       *)
      validateTemplate[t];
      formatTemplate@NormalizeTemplate[t]
    ]

formatTemplate[template_] :=
    Block[{LFun, LOFun, LClass, LTemplate, LType, LExpressionID},
      With[{tem = template},
        LType /: {LType[head_, rest___], passing : passingMethodPattern} :=
            If[{passing} =!= {}, passing <> " ", ""] <>
            ToString[head] <>
            If[{rest} =!= {}, "<" <> StringTake[ToString[{rest}], {2,-2}] <> ">", ""];
        LExpressionID[head_String] := "LExpressionID<" <> head <> ">";
        LFun[name_, args_, ret_] := StringTemplate["`` ``(``)"][ToString[ret], name, StringJoin@Riffle[ToString /@ args, ", "]];
        LOFun[name_] := StringTemplate["LinkObject ``(LinkObject)"][name];
        LClass[name_, funs_] := StringTemplate["class ``:\n``"][name, StringJoin@Riffle["    " <> ToString[#] & /@ funs, "\n"]];
        LTemplate[name_, classes_] := StringTemplate["template ``\n\n"][name] <> Riffle[ToString /@ classes, "\n\n"];
        tem
      ]
    ]


End[] (* End Private Context *)
