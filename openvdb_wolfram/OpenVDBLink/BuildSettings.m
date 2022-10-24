(* ::Package:: *)

(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


Switch[$OperatingSystem,
    "MacOSX",
        $buildSettings = {
            "CompileOptions" -> {"-std=c++17 -ltbb -lopenvdb -flto"},
            "Compiler" -> CCompilerDriver`ClangCompiler`ClangCompiler
        };
        $libraryName = "OpenVDBLink.dylib",
    "Windows",
        $vcpkgDir = $HomeDirectory; (* change this to the location of vcpkg *)
        $vcpkgInstalled = FileNameJoin[{$vcpkgDir, "vcpkg", "installed", "x64-windows"}];
        $vcpkgLib = FileNameJoin[{$vcpkgInstalled, "lib"}];

        $buildSettings = {
            "CompileOptions" -> {"/std:c++17", "/EHsc", "/GL", "/wd4244", "/DNOMINMAX"},
            "Compiler" -> CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler,
            "IncludeDirectories" -> {
                FileNameJoin[{"C:", "Program Files", "OpenVDB", "include"}],
                FileNameJoin[{$vcpkgInstalled, "include"}],
                FileNameJoin[{$vcpkgDir, "vcpkg", "packages", "boost-numeric-conversion_x64-windows", "include"}]
            },
            "ExtraObjectFiles" -> Join[
                {
                    FileNameJoin[{$vcpkgDir, "openvdb", "build", "openvdb", "openvdb", "Release", "libopenvdb.lib"}],
                    FileNameJoin[{$vcpkgLib, "blosc.lib"}],
                    FileNameJoin[{$vcpkgLib, "zlib.lib"}]
                },
                FileNames[FileNameJoin[{$vcpkgLib, "tbb*.lib"}]]
            ]
        };
        $libraryName = "OpenVDBLink.dll",
    "Unix",
        $buildSettings = {
            "CompileOptions" -> {"-std=c++17 -ltbb -lopenvdb -flto"},
            "Compiler" -> CCompilerDriver`GCCCompiler`GCCCompiler,
            "CompilerName" -> "g++"
        };
        $libraryName = "OpenVDBLink.so",
    _,
        $buildSettings = None;
        $libraryName = ""
];
