(* ::Package:: *)

Switch[$OperatingSystem,
    "MacOSX",
        $buildSettings = {
            "CompileOptions" -> {"-std=c++14 -ltbb -lHalf -lopenvdb -flto"},
            "Compiler" -> CCompilerDriver`ClangCompiler`ClangCompiler
        };
        $libraryName = "OpenVDBLink.dylib",
    "Windows",
        $vcpkgDir = $HomeDirectory; (* change this to the location of vcpkg *)
        $vcpkgInstalled = FileNameJoin[{$vcpkgDir, "vcpkg", "installed", "x64-windows"}];
        $vcpkgLib = FileNameJoin[{$vcpkgInstalled, "lib"}];
        
        $buildSettings = {
            "CompileOptions" -> {"/EHsc", "/GL", "/wd4244", "/DNOMINMAX"},
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
    _, (* TODO "Unix", etc. *)
        $buildSettings = None;
        $libraryName = ""
];
