(* ::Package:: *)

If[$CloudEvaluation,
	Print["OpenVDBLink cannot run in the Wolfram Cloud because the cloud does not support LibraryLink.  Aborting."];
	Abort[]
];


Unprotect["OpenVDBLink`*"];


Get["OpenVDBLink`OpenVDBLink`"];


With[{$userSyms = {"$OpenVDBCreator", "$OpenVDBHalfWidth", "$OpenVDBSpacing"}},
	Scan[SetAttributes[#, {Protected, ReadProtected}]&, Complement[Names["OpenVDBLink`*"], $userSyms]];
]
